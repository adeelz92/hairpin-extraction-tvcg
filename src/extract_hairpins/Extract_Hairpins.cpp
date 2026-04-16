#include <vtkActor.h>
#include <vtkAppendFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCamera.h>
#include <vtkCellData.h>
#include <vtkCellCenters.h>
#include <vtkClosestPointStrategy.h>
#include <vtkCompositeInterpolatedVelocityField.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkHausdorffDistancePointSetFilter.h>
#include <vtkStaticCellLocator.h>
#include <vtkStaticPointLocator.h>
#include <vtkCellLocator.h>
#include <vtkContourFilter.h>
#include <vtkPointLocator.h>
#include <vtkCleanPolyData.h>
#include <vtkColorTransferFunction.h>
#include <vtkConnectivityFilter.h>
#include <vtkCubeAxesActor.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkFillHolesFilter.h>
#include <vtkFloatArray.h>
#include <vtkGaussianKernel.h>
#include <vtkVoronoiKernel.h>
#include <vtkGeometryFilter.h>
#include <vtkInformation.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLine.h>
#include <vtkLookupTable.h>
#include <vtkMaskPoints.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkOutlineFilter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataConnectivityFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLine.h>
#include <vtkPointInterpolator.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarBarActor.h>
#include <vtkSelectEnclosedPoints.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkSMPTools.h>
#include <vtkSphereSource.h>
#include <vtkStreamTracer.h>
#include <vtkSTLWriter.h>
#include <vtkStripper.h>
#include <vtkTextProperty.h>
#include <vtkTriangleFilter.h>
#include <vtkThreshold.h>
#include <vtkThresholdPoints.h>
#include <vtkUnstructuredGrid.h>

#include <ttkGeometrySmoother.h>
#include <ttkManifoldCheck.h>

#include <random>
#include <set>
#include <unordered_set>
#include <future>
#include <chrono>
#include <thread>
#include <fstream>
#include <vtkMinimalStandardRandomSequence.h>

vtkSmartPointer<vtkDataSet> DATASET;
vtkSmartPointer<vtkDataSet> ISOSURFACES;
std::unordered_set<vtkIdType> procesdRegions;
std::unordered_set<vtkIdType> procesdHeads;
std::unordered_set<vtkIdType> visitedRegions;
std::unordered_set<std::string> regionStrings;
std::unordered_map<vtkIdType, vtkSmartPointer<vtkUnstructuredGrid>> regionsMap;
std::unordered_map<vtkIdType, vtkSmartPointer<vtkUnstructuredGrid>> segmentsMap;
std::unordered_map<vtkIdType, vtkSmartPointer<vtkPolyData>> isosurfacesMap;
vtkSmartPointer<vtkAbstractCellLocator> CellLocator;
vtkSmartPointer<vtkAbstractCellLocator> isosurfacesCellLocator;
vtkSmartPointer<vtkAbstractPointLocator> PointLocator;

float streamTracerTime = 0;
float findCandidatesTime = 0;
float skelExtractionTime = 0;
float surfaceExtractionTime = 0;
float postProcessingTime = 0;
float skelFindTime = 0;

float interpolatorTime = 0;
float cleanTime = 0;
float smoothFilterTime = 0;
float CONSISTENCY = 0.8;

template <typename T>
struct Region
{
	int RegionId;
	vtkSmartPointer<T> RegionData;

	Region(int regionId, T* regionData)
	{
		this->RegionId = regionId;
		this->RegionData = vtkSmartPointer<T>::New();
		this->RegionData->DeepCopy(regionData);
	}

	void updateData(T* regionData)
	{
		this->RegionData->DeepCopy(regionData);
	}
};


//////////Extract Skeleton Functor/////////////
#define CGAL_EIGEN3_ENABLED
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

typedef CGAL::Simple_cartesian<double>                        Kernel;
typedef CGAL::Polyhedron_3<Kernel>                            Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;

float Quality_Speed_Tradeoff = 0.2;
float Medially_centered_speed_tradeoff = 0.1;
bool contract_with_timeout(Skeletonization& mcs, double timeout_seconds = 60.0) {
	std::atomic<bool> timeout_occurred{ false };

	// Start skeletonization asynchronously
	std::future<void> future = std::async(std::launch::async, [&]() {
		mcs.contract_until_convergence();
		});

	auto start_time = std::chrono::steady_clock::now();

	// Keep checking if the function has finished
	while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
		auto elapsed = std::chrono::steady_clock::now() - start_time;
		if (std::chrono::duration<double>(elapsed).count() >= timeout_seconds) {
			std::cout << "Timeout! Stopping skeletonization.\n";
			mcs.set_max_iterations(1); // Force it to stop ASAP
			timeout_occurred = true;
			break;
		}
	}

	// Ensure the function completes properly
	future.wait();

	return !timeout_occurred;
}

void readSkelInfo(std::string& filename, vtkPolyData* skeleton)
{
	fstream fin;
	fin.open(filename.c_str(), std::ios::in);
	if (!fin.is_open()) throw std::runtime_error("Could not open file");

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	std::string line;
	double val, numPts;
	vtkSmartPointer<vtkPolyLine> polyLine = nullptr;

	int ptIdx = -1;
	int linePtIdx;
	while (std::getline(fin, line))
	{
		std::stringstream ss(line);
		std::vector<double> results;

		while (ss >> val) {
			results.push_back(val);
		}

		if (results.size() == 1)
		{
			linePtIdx = 0;
			numPts = results.at(0);
			// cout << numPts << endl;
			if (polyLine != nullptr)
			{
				cells->InsertNextCell(polyLine);
			}
			polyLine = vtkSmartPointer<vtkPolyLine>::New();
			polyLine->GetPointIds()->SetNumberOfIds(numPts);
		}

		if (results.size() > 1)
		{
			double pt[3] = { results.at(0), results.at(1), results.at(2) };
			/**/
			// cout << results.at(0) << " " << results.at(1) << " " << results.at(2);
			bool found = false;
			int i;
			for (i = 0; i < points->GetNumberOfPoints(); i++)
			{
				double x[3];
				points->GetPoint(i, x);
				x[0] = x[0]; x[1] = x[1]; x[2] = x[2];
				if (pt[0] == x[0] && pt[1] == x[1] && pt[2] == x[2])
				{
					found = true;
					break;
				}
			}
			// cout << " " << polyLine->GetNumberOfPoints() << endl;
			if (!found)
			{
				ptIdx++;
				points->InsertNextPoint(pt);
				polyLine->GetPointIds()->SetId(linePtIdx, ptIdx);
				linePtIdx++;
			}
			else
			{
				polyLine->GetPointIds()->SetId(linePtIdx, i);
				linePtIdx++;
			}
		}
	}
	cells->InsertNextCell(polyLine);

	// Add the points to the dataset
	skeleton->SetPoints(points);
	// Add the lines to the dataset
	skeleton->SetLines(cells);
	fin.close();
}

//only needed for the display of the skeleton as maximal polylines
struct Display_polylines {
	const Skeleton& skeleton;
	std::ofstream& out;
	int polyline_size;
	std::stringstream sstr;
	Display_polylines(const Skeleton& skeleton, std::ofstream& out)
		: skeleton(skeleton), out(out)
	{
	}
	void start_new_polyline() {
		polyline_size = 0;
		sstr.str("");
		sstr.clear();
	}
	void add_node(Skeleton_vertex v) {
		++polyline_size;
		sstr << skeleton[v].point << "\n";
	}
	void end_polyline()
	{
		out << polyline_size << "\n" << sstr.str();
	}
};

void refineSkeleton(vtkPolyData* skeleton)
{
	// Remove vortex cells (ioslated points)
	int numCells = skeleton->GetNumberOfCells();
	for (int i = 0; i < numCells; i++)
	{
		vtkCell* polyLine = skeleton->GetCell(i);
		if (polyLine->GetCellType() == 0 || polyLine->GetCellType() == 1) //remove vertex cells
		{
			skeleton->DeleteCell(i);
		}
	}
	skeleton->RemoveDeletedCells();
}

// This example extracts a medially centered skeleton from a given mesh.
bool getSkeleton(const std::string filename, const std::string output_filename)
{
	Polyhedron tmesh;
	if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(filename, tmesh))
	{
		std::cerr << "Invalid input." << std::endl;
		return false;
	}

	Skeleton skeleton;
	try {
		Skeletonization mcs(tmesh);
		// CGAL::extract_mean_curvature_flow_skeleton(tmesh, skeleton);
		mcs.set_quality_speed_tradeoff(Quality_Speed_Tradeoff);
		mcs.set_medially_centered_speed_tradeoff(Medially_centered_speed_tradeoff);
		// Run with timeout
		if (!contract_with_timeout(mcs, 60.0)) {
			return false;  // Timeout occurred 
		}
		mcs.convert_to_skeleton(skeleton);
	}
	catch (const std::exception& e) {
		return false;
	}

	// std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
	// std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";
	// Output all the edges of the skeleton.
	std::ofstream output(output_filename);
	Display_polylines display(skeleton, output);
	CGAL::split_graph_into_polylines(skeleton, display);
	output.close();

	return true;
}

//////////End Extract Skeleton Functor/////////////

void RemoveCycles(vtkPolyData* inputPolyData)
{
	// Step 1: Build adjacency map
	std::unordered_map<vtkIdType, std::unordered_set<vtkIdType>> adjacency;
	auto lines = inputPolyData->GetLines();
	lines->InitTraversal();

	vtkIdType npts;
	const vtkIdType* pts;
	while (lines->GetNextCell(npts, pts))
	{
		if (npts != 2)
			continue; // only process 2-point lines
		vtkIdType p0 = pts[0];
		vtkIdType p1 = pts[1];
		adjacency[p0].insert(p1);
		adjacency[p1].insert(p0);
	}

	// Step 2: DFS to detect cycles
	std::unordered_set<vtkIdType> visited;
	std::unordered_set<vtkIdType> recStack;
	std::set<std::pair<vtkIdType, vtkIdType>> edgesToRemove;

	std::function<void(vtkIdType, vtkIdType)> dfs = [&](vtkIdType current, vtkIdType parent)
		{
			visited.insert(current);
			recStack.insert(current);

			for (auto neighbor : adjacency[current])
			{
				if (neighbor == parent)
					continue;

				if (recStack.find(neighbor) != recStack.end())
				{
					// Cycle detected: mark this edge
					edgesToRemove.insert(std::minmax(current, neighbor));
				}
				else if (visited.find(neighbor) == visited.end())
				{
					dfs(neighbor, current);
				}
			}

			recStack.erase(current);
		};

	// Start DFS from all unvisited nodes
	for (const auto& kv : adjacency)
	{
		vtkIdType pointId = kv.first;
		if (visited.find(pointId) == visited.end())
		{
			dfs(pointId, -1);
		}
	}

	// Step 3: Create new CellArray without edges that close cycles
	auto newLines = vtkSmartPointer<vtkCellArray>::New();
	lines->InitTraversal();
	while (lines->GetNextCell(npts, pts))
	{
		if (npts != 2)
			continue;

		vtkIdType p0 = pts[0];
		vtkIdType p1 = pts[1];
		auto e = std::minmax(p0, p1);

		if (edgesToRemove.find(e) == edgesToRemove.end())
		{
			// Keep this edge
			newLines->InsertNextCell(2, pts);
		}
	}

	inputPolyData->SetLines(newLines);
	inputPolyData->Modified();
}

void RemoveCycles2(vtkPolyData* inputPolyData)
{
	// Step 1: Build adjacency map
	std::unordered_map<vtkIdType, std::unordered_set<vtkIdType>> adjacency;
	auto lines = inputPolyData->GetLines();
	lines->InitTraversal();

	vtkIdType npts;
	const vtkIdType* pts;
	while (lines->GetNextCell(npts, pts))
	{
		if (npts != 2)
			continue; // only process 2-point lines
		vtkIdType p0 = pts[0];
		vtkIdType p1 = pts[1];
		adjacency[p0].insert(p1);
		adjacency[p1].insert(p0);
	}

	// Step 2: DFS to detect cycles
	std::unordered_set<vtkIdType> visited;
	std::unordered_set<vtkIdType> recStack;
	std::set<std::pair<vtkIdType, vtkIdType>> edgesToRemove;

	std::function<void(vtkIdType, vtkIdType)> dfs = [&](vtkIdType current, vtkIdType parent)
		{
			visited.insert(current);
			recStack.insert(current);

			for (auto neighbor : adjacency[current])
			{
				if (neighbor == parent)
					continue;

				if (recStack.find(neighbor) != recStack.end())
				{
					// Cycle detected: mark this edge
					edgesToRemove.insert(std::minmax(current, neighbor));
				}
				else if (visited.find(neighbor) == visited.end())
				{
					dfs(neighbor, current);
				}
			}

			recStack.erase(current);
		};

	// Start DFS from all unvisited nodes
	for (const auto& kv : adjacency)
	{
		vtkIdType pointId = kv.first;
		if (visited.find(pointId) == visited.end())
		{
			dfs(pointId, -1);
		}
	}

	// Step 3: Compute total edge length and cycle edge length
	auto points = inputPolyData->GetPoints();
	double totalLength = 0.0;
	double cycleLength = 0.0;

	lines->InitTraversal();
	while (lines->GetNextCell(npts, pts))
	{
		if (npts != 2)
			continue;

		double p0[3], p1[3];
		points->GetPoint(pts[0], p0);
		points->GetPoint(pts[1], p1);

		double len = std::sqrt(vtkMath::Distance2BetweenPoints(p0, p1));
		totalLength += len;

		auto e = std::minmax(pts[0], pts[1]);
		if (edgesToRemove.find(e) != edgesToRemove.end())
		{
			cycleLength += len;
		}
	}

	// Step 4: Only remove cycles if they are < 1% of total length
	bool removeCycles = (totalLength > 0.0) && (cycleLength / totalLength < 0.05);

	// Step 5: Create new CellArray with condition
	auto newLines = vtkSmartPointer<vtkCellArray>::New();
	lines->InitTraversal();
	while (lines->GetNextCell(npts, pts))
	{
		if (npts != 2)
			continue;

		auto e = std::minmax(pts[0], pts[1]);
		bool isCycleEdge = (edgesToRemove.find(e) != edgesToRemove.end());

		if (!removeCycles || !isCycleEdge)
		{
			newLines->InsertNextCell(2, pts);
		}
	}

	inputPolyData->SetLines(newLines);
	inputPolyData->Modified();
}

void extractSkeleton(vtkPolyData* surface, vtkPolyData* skel)
{

	std::string filename = "mesh";
	std::string ext = ".stl";
	std::string mesh_filename_wExt = filename + ext;
	// std::string mesh_filepath = "mesh_fixed.stl";

	// First write the extracted surface to an STL file
	vtkSmartPointer<vtkSTLWriter> stlWriter = vtkSmartPointer<vtkSTLWriter>::New();
	stlWriter->SetFileName(mesh_filename_wExt.c_str());
	stlWriter->SetInputData(surface);
	stlWriter->SetFileTypeToBinary();
	stlWriter->Write();

	std::string skel_output_path = filename + ".txt";
	bool success = getSkeleton(mesh_filename_wExt, skel_output_path);
	// bool success = getSkeletonFromVTK(surface, skel_output_path);

	if (!success)
	{
		return;
	}

	readSkelInfo(skel_output_path, skel);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vtkSmartPointer<vtkCleanPolyData> clean = vtkSmartPointer<vtkCleanPolyData>::New();
	clean->SetInputData(skel);
	clean->PointMergingOn();
	clean->Update();

	vtkSmartPointer<vtkStripper> stripper = vtkSmartPointer<vtkStripper>::New();
	stripper->SetInputData(clean->GetOutput());
	stripper->JoinContiguousSegmentsOn();
	stripper->Update();

	if (stripper->GetOutput()->GetNumberOfCells() == 0) {
		return;
	}

	vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
	smoothFilter->SetInputData(stripper->GetOutput());
	smoothFilter->SetNumberOfIterations(25);
	smoothFilter->SetRelaxationFactor(0.75);
	smoothFilter->BoundarySmoothingOn();
	smoothFilter->SetEdgeAngle(270);
	smoothFilter->Update();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	smoothFilterTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	refineSkeleton(smoothFilter->GetOutput());

	/*
	vtkSmartPointer<vtkPolyDataConnectivityFilter> connFilter1 = vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	connFilter1->SetInputData(smoothFilter->GetOutput());
	connFilter1->SetExtractionModeToLargestRegion(); // We only need to check one largest spanwise component
	connFilter1->ColorRegionsOff();
	connFilter1->Update();
	*/

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(smoothFilter->GetOutput());
	triangleFilter->Update();

	skel->DeepCopy(triangleFilter->GetOutput());
	// RemoveCycles(skel);
}

void RandomColors(vtkLookupTable* lut, int numberOfColors)
{
	// Fill in a few known colors, the rest will be generated if needed
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
	lut->SetTableValue(0, colors->GetColor4d("Black").GetData());
	lut->SetTableValue(1, colors->GetColor4d("Banana").GetData());
	lut->SetTableValue(2, colors->GetColor4d("Tomato").GetData());
	lut->SetTableValue(3, colors->GetColor4d("Wheat").GetData());
	lut->SetTableValue(4, colors->GetColor4d("Lavender").GetData());
	lut->SetTableValue(5, colors->GetColor4d("Flesh").GetData());
	lut->SetTableValue(6, colors->GetColor4d("Raspberry").GetData());
	lut->SetTableValue(7, colors->GetColor4d("Salmon").GetData());
	lut->SetTableValue(8, colors->GetColor4d("Mint").GetData());
	lut->SetTableValue(9, colors->GetColor4d("Peacock").GetData());

	// If the number of colors is larger than the number of specified colors,
	// generate some random colors.
	vtkSmartPointer<vtkMinimalStandardRandomSequence> randomSequence =
		vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
	randomSequence->SetSeed(4355412);
	if (numberOfColors > 9)
	{
		for (auto i = 10; i < numberOfColors; ++i)
		{
			double r, g, b;
			r = randomSequence->GetRangeValue(0.6, 1.0);
			randomSequence->Next();
			g = randomSequence->GetRangeValue(0.6, 1.0);
			randomSequence->Next();
			b = randomSequence->GetRangeValue(0.6, 1.0);
			randomSequence->Next();
			lut->SetTableValue(i, r, g, b, 1.0);
		}
	}
}

void findEndPointsTriangle(vtkPolyData* skeleton, vtkIdType& minPtId, vtkSmartPointer<vtkIdList> endPointIds)
{
	double pt[3];
	float minX = std::numeric_limits<float>::max();
	vtkSmartPointer<vtkIdList> cellPtIds = vtkSmartPointer<vtkIdList>::New();
	vtkSmartPointer<vtkIdList> ptCellIds = vtkSmartPointer<vtkIdList>::New();
	for (int i = 0; i < skeleton->GetNumberOfCells(); i++)
	{
		skeleton->GetCellPoints(i, cellPtIds);
		// Check the first point
		skeleton->GetPoint(cellPtIds->GetId(0), pt);
		if (pt[0] < minX) {
			minX = pt[0];
			minPtId = cellPtIds->GetId(0);
		}
		skeleton->GetPointCells(cellPtIds->GetId(0), ptCellIds);
		if (ptCellIds->GetNumberOfIds() == 1) {
			endPointIds->InsertNextId(cellPtIds->GetId(0));
		}
		// Check the second point
		skeleton->GetPoint(cellPtIds->GetId(cellPtIds->GetNumberOfIds() - 1), pt);
		if (pt[0] < minX) {
			minX = pt[0];
			minPtId = cellPtIds->GetId(cellPtIds->GetNumberOfIds() - 1);
		}
		skeleton->GetPointCells(cellPtIds->GetId(cellPtIds->GetNumberOfIds() - 1), ptCellIds);
		if (ptCellIds->GetNumberOfIds() == 1) {
			endPointIds->InsertNextId(cellPtIds->GetId(cellPtIds->GetNumberOfIds() - 1));
		}
	}
}

bool inspectRegion(vtkDataSet* dataset)
{
	// Here we check of the avg oyf of the region is greater than 0?
	vtkDataArray* oyfArray = dataset->GetPointData()->GetArray("oyf");
	double oyfAvg = 0.0;
	for (vtkIdType tupleId = 0; tupleId < oyfArray->GetNumberOfTuples(); tupleId++)
	{
		oyfAvg += oyfArray->GetTuple1(tupleId);
	}
	oyfAvg /= oyfArray->GetNumberOfTuples();

	if (oyfAvg > 0)
	{
		return true;
	}

	return false;
}

void createLookupTable(vtkLookupTable* lut, vtkNamedColors* colors, int numberOfRegions)
{
	lut->SetNumberOfTableValues(std::max(numberOfRegions + 1, 10));
	lut->Build();
	lut->SetTableValue(0, colors->GetColor4d("Black").GetData());
	lut->SetTableValue(1, colors->GetColor4d("MediumVioletRed").GetData());
	lut->SetTableValue(2, colors->GetColor4d("OrangeRed").GetData());
	lut->SetTableValue(3, colors->GetColor4d("DarkKhaki").GetData());
	lut->SetTableValue(4, colors->GetColor4d("Indigo").GetData());
	lut->SetTableValue(5, colors->GetColor4d("DarkGreen").GetData());
	lut->SetTableValue(6, colors->GetColor4d("DarkBlue").GetData());
	lut->SetTableValue(7, colors->GetColor4d("SaddleBrown").GetData());
	lut->SetTableValue(8, colors->GetColor4d("AntiqueWhite").GetData());
	lut->SetTableValue(9, colors->GetColor4d("Gray").GetData());

	if (numberOfRegions > 9)
	{
		std::mt19937 mt(4355412); // Standard mersenne_twister_engine
		std::uniform_real_distribution<double> distribution(.1, 0.5);
		for (auto i = 10; i < numberOfRegions; ++i)
		{
			lut->SetTableValue(i, lut->GetTableValue(i % 9));
		}
	}

	lut->SetTableValue(numberOfRegions, colors->GetColor4d("DarkRed").GetData());
}

void extractSurface(vtkDataSet* dataset, vtkPolyData* output)
{
	if (dataset->GetNumberOfPoints() == 0 || dataset->GetNumberOfCells() == 0)
	{
		output = nullptr;
		return;
	}

	vtkSmartPointer<vtkGeometryFilter> surfaceFilter = vtkSmartPointer<vtkGeometryFilter>::New();
	surfaceFilter->SetInputData(dataset);
	surfaceFilter->Update();

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(surfaceFilter->GetOutput());
	triangleFilter->Update();

	vtkSmartPointer<ttkGeometrySmoother> smooth = vtkSmartPointer<ttkGeometrySmoother>::New();
	smooth->SetInputData(triangleFilter->GetOutput());
	smooth->SetNumberOfIterations(10);
	smooth->Update();

	vtkSmartPointer<vtkPolyDataConnectivityFilter> surfaceConnectivity =
		vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	surfaceConnectivity->SetInputData(smooth->GetOutput());
	surfaceConnectivity->ColorRegionsOff();
	surfaceConnectivity->SetExtractionModeToLargestRegion();
	surfaceConnectivity->Update();

	output->DeepCopy(surfaceConnectivity->GetOutput());

}

// Distance-based isosurface clearning strategy.
void extractIsosurface(vtkPolyData* surface, vtkPolyData* isosurface) {

	vtkSmartPointer<vtkCellCenters> surfaceCellCenters = vtkSmartPointer<vtkCellCenters>::New();
	surfaceCellCenters->SetInputData(surface);
	surfaceCellCenters->Update();
	vtkPolyData* surfaceCenters = surfaceCellCenters->GetOutput();

	std::unordered_set<int> uniqueColorIds;
	vtkDataArray* surfaceColorIds = ISOSURFACES->GetCellData()->GetArray("SurfaceRegionIds");

	/*
	double pt[3], closestPoint[3];
	vtkIdType cellId;
	int subId, colorId;
	double dist2;
	for (int i = 0; i < surfaceCenters->GetNumberOfPoints(); ++i) {
		surfaceCenters->GetPoint(i, pt);
		isosurfacesCellLocator->FindClosestPoint(pt, closestPoint, cellId, subId, dist2);
		colorId = surfaceColorIds->GetTuple1(cellId);
		uniqueColorIds.insert(colorId);
	}
	*/

	vtkSMPTools::SetBackend("STDThread");
	// Thread-local storage
	vtkSMPThreadLocal<std::unordered_set<int>> threadLocalSets;
	auto SurfaceColorIds = vtk::DataArrayTupleRange<1>(surfaceColorIds);
	double test_pt[3];
	surfaceCenters->GetPoint(0, test_pt);

	vtkSMPTools::For(0, surfaceCenters->GetNumberOfPoints(), [&](vtkIdType start, vtkIdType end) {

		auto& localSet = threadLocalSets.Local();  // Each thread gets its own map
		double point[3], closestPoint[3];
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New(); // thread-local cell
		vtkIdType cellId;
		int subId;
		double dist2;

		for (vtkIdType j = start; j < end; ++j) {
			surfaceCenters->GetPoint(j, point);

			isosurfacesCellLocator->FindClosestPoint(
				point, closestPoint, localCell, cellId, subId, dist2);

			if (cellId >= 0) {
				vtkIdType regId = static_cast<vtkIdType>(SurfaceColorIds[cellId][0]);
				localSet.insert(regId);
			}
		}
		});

	// Merge results
	for (auto& s : threadLocalSets)
	{
		uniqueColorIds.insert(s.begin(), s.end());
	}
	vtkSMPTools::SetBackend("Sequential");
	/*
	vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();
	cellLocator->SetDataSet(surface);
	cellLocator->BuildLocator();

	std::unordered_map<int, double> avgColorDists;
	for (int colorId : uniqueColorIds) {
		vtkPolyData* isosurface_seg = isosurfacesMap[colorId];
		for (int cid = 0; cid < isosurface_seg->GetNumberOfCells(); ++cid) {
			double center[3] = { 0, 0, 0 };
			vtkCell* cell = isosurface_seg->GetCell(cid);
			double pt[3];
			for (vtkIdType i = 0; i < cell->GetNumberOfPoints(); ++i) {
				isosurface_seg->GetPoint(cell->GetPointId(i), pt);
				center[0] += pt[0];
				center[1] += pt[1];
				center[2] += pt[2];
			}
			center[0] /= cell->GetNumberOfPoints();
			center[1] /= cell->GetNumberOfPoints();
			center[2] /= cell->GetNumberOfPoints();

			double closestPoint[3];
			vtkIdType cellId;
			int subId;
			double dist2;
			cellLocator->FindClosestPoint(center, closestPoint, cellId, subId, dist2);
			avgColorDists[colorId] += dist2;
		}
		avgColorDists[colorId] /= isosurface_seg->GetNumberOfCells();
	}

	double avgDist = 0.0;
	for (auto pair : avgColorDists) {
		avgDist += pair.second;
	}
	avgDist /= avgColorDists.size();
	*/

	vtkSmartPointer<vtkAppendFilter> appendPDFilter = vtkSmartPointer<vtkAppendFilter>::New();
	appendPDFilter->MergePointsOn();
	for (int colorId : uniqueColorIds) {
		/*
		if (avgColorDists[colorId] > avgDist) {
			continue;
		}
		*/
		appendPDFilter->AddInputData(isosurfacesMap[colorId]);
	}
	appendPDFilter->Update();

	vtkSmartPointer<vtkGeometryFilter> surfaceFilter = vtkSmartPointer<vtkGeometryFilter>::New();
	surfaceFilter->SetInputData(appendPDFilter->GetOutput());
	surfaceFilter->Update();

	isosurface->DeepCopy(surfaceFilter->GetOutput());
}

void addTypeArray(vtkPolyData* pd, const int& type)
{
	vtkSmartPointer<vtkIntArray> typeArray = vtkSmartPointer<vtkIntArray>::New();
	typeArray->SetNumberOfComponents(1);
	typeArray->SetNumberOfTuples(pd->GetNumberOfCells());
	typeArray->SetName("VortexType");
	for (int j = 0; j < pd->GetNumberOfCells(); j++)
	{
		typeArray->SetTuple1(j, type);
	}
	pd->GetCellData()->AddArray(typeArray);
}

void getNewSurface(vtkDataSet* jointRegions, vtkPolyData* newSurface) {
	vtkSmartPointer<vtkGeometryFilter> surfaceFilter = vtkSmartPointer<vtkGeometryFilter>::New();
	surfaceFilter->SetInputData(jointRegions);
	surfaceFilter->Update();

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(surfaceFilter->GetOutput());
	triangleFilter->Update();

	vtkSmartPointer<ttkGeometrySmoother> smooth = vtkSmartPointer<ttkGeometrySmoother>::New();
	smooth->SetInputData(triangleFilter->GetOutput());
	smooth->SetNumberOfIterations(10);
	smooth->Update();

	vtkSmartPointer<vtkPolyDataConnectivityFilter> connFil = vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	connFil->SetInputData(smooth->GetOutput());
	connFil->SetExtractionModeToLargestRegion();
	connFil->ColorRegionsOff();
	connFil->Update();

	newSurface->DeepCopy(connFil->GetOutput());
}

void fillHoles(vtkPolyData* isosurface, bool smooth = false) {

	vtkSmartPointer<vtkPolyDataConnectivityFilter> larConnFil =
		vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	larConnFil->SetInputData(isosurface);
	larConnFil->SetExtractionModeToLargestRegion();
	larConnFil->ColorRegionsOff();
	larConnFil->Update();

	/**/
	vtkSmartPointer<vtkFillHolesFilter> fillHolesFilter = vtkSmartPointer<vtkFillHolesFilter>::New();
	fillHolesFilter->SetInputData(larConnFil->GetOutput());
	fillHolesFilter->SetHoleSize(100000.0);
	fillHolesFilter->Update();

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(fillHolesFilter->GetOutput());
	triangleFilter->Update();
	isosurface->DeepCopy(triangleFilter->GetOutput());
	if (isosurface->GetNumberOfPoints() == 0 || isosurface->GetNumberOfCells() == 0) {
		isosurface = nullptr;
		return;
	}

	if (smooth) {
		vtkSmartPointer<ttkGeometrySmoother> smoothFilter = vtkSmartPointer<ttkGeometrySmoother>::New();
		smoothFilter->SetInputData(triangleFilter->GetOutput());
		smoothFilter->SetNumberOfIterations(2);
		smoothFilter->Update();
		isosurface->DeepCopy(smoothFilter->GetOutput());
	}
}

void constructSurface(vtkDataSet* jointRegions, vtkPolyData* isosurface, vtkPolyData* newSurface)
{
	// Get disconnected components of the isosurface
	vtkSmartPointer<vtkConnectivityFilter> isoConnFil =
		vtkSmartPointer<vtkConnectivityFilter>::New();
	isoConnFil->SetInputData(isosurface);
	isoConnFil->SetExtractionModeToAllRegions();
	isoConnFil->ColorRegionsOn();
	isoConnFil->Update();

	if (isoConnFil->GetNumberOfExtractedRegions() == 1) {
		return;
	}

	vtkDataSet* isosurfaceDataset = isoConnFil->GetOutput();
	vtkPointSet* PS = vtkPointSet::SafeDownCast(isosurfaceDataset);
	PS->BuildCellLocator();
	PS->EditableOff();

	vtkSmartPointer<vtkAbstractCellLocator> cellLocator = PS->GetCellLocator();
	vtkDataArray* regionIdArr = isosurfaceDataset->GetCellData()->GetArray("RegionId");

	vtkSmartPointer<vtkCellCenters> cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(jointRegions);
	cellCenters->Update();

	vtkDataSet* centers = cellCenters->GetOutput();

	vtkSmartPointer<vtkIntArray> colorIdsArray = vtkSmartPointer<vtkIntArray>::New();
	colorIdsArray->SetNumberOfComponents(1);
	colorIdsArray->SetNumberOfTuples(jointRegions->GetNumberOfCells());
	colorIdsArray->SetName("ColorIds");

	std::unordered_map<int, int> colorIdsCount;
	double pt[3], closestPoint[3];
	vtkIdType cellId;
	int subId, colorId;
	double dist2;
	for (int i = 0; i < jointRegions->GetNumberOfCells(); ++i) {
		centers->GetPoint(i, pt);
		cellLocator->FindClosestPoint(pt, closestPoint, cellId, subId, dist2);
		colorId = regionIdArr->GetTuple1(cellId);
		colorIdsArray->SetTuple1(i, colorId);
		colorIdsCount[colorId]++;
	}
	jointRegions->GetCellData()->AddArray(colorIdsArray);

	int maxColorIdsCount = 0;
	int maxColorId = -1;
	for (auto pair : colorIdsCount) {
		if (pair.second > maxColorIdsCount) {
			maxColorIdsCount = pair.second;
			maxColorId = pair.first;
		}
	}

	vtkSmartPointer<vtkThreshold> threshold1 = vtkSmartPointer<vtkThreshold>::New();
	threshold1->SetInputData(jointRegions);
	threshold1->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "ColorIds");
	threshold1->SetLowerThreshold(maxColorId);
	threshold1->SetUpperThreshold(maxColorId);
	threshold1->Update();
	jointRegions->DeepCopy(threshold1->GetOutput());

	vtkSmartPointer<vtkGeometryFilter> surfaceFilter = vtkSmartPointer<vtkGeometryFilter>::New();
	surfaceFilter->SetInputData(threshold1->GetOutput());
	surfaceFilter->Update();

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(surfaceFilter->GetOutput());
	triangleFilter->Update();

	vtkSmartPointer<ttkGeometrySmoother> smooth = vtkSmartPointer<ttkGeometrySmoother>::New();
	smooth->SetInputData(triangleFilter->GetOutput());
	smooth->SetNumberOfIterations(10);
	smooth->Update();

	vtkSmartPointer<vtkPolyDataConnectivityFilter> connFil = vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	connFil->SetInputData(smooth->GetOutput());
	connFil->SetExtractionModeToLargestRegion();
	connFil->ColorRegionsOff();
	connFil->Update();

	newSurface->DeepCopy(connFil->GetOutput());

	vtkSmartPointer<vtkPolyDataConnectivityFilter> larConnFil =
		vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	larConnFil->SetInputData(isosurface);
	larConnFil->SetExtractionModeToLargestRegion();
	larConnFil->ColorRegionsOff();
	larConnFil->Update();
	isosurface->DeepCopy(larConnFil->GetOutput());

	/*
	vtkSmartPointer<vtkFillHolesFilter> fillHolesFilter = vtkSmartPointer<vtkFillHolesFilter>::New();
	fillHolesFilter->SetInputData(larConnFil->GetOutput());
	fillHolesFilter->SetHoleSize(100000.0);
	fillHolesFilter->Update();
	*/

	jointRegions->GetCellData()->RemoveArray("ColorIds");
	newSurface->GetCellData()->RemoveArray("ColorIds");
}

double round_up(double value, int decimal_places)
{
	const double multiplier = std::pow(10.0, decimal_places);
	return std::ceil(value * multiplier) / multiplier;
}

vtkSmartPointer<vtkCellArray> SeparatePolyLineBranches(vtkPolyData* inputPolyData) {
	vtkSmartPointer<vtkCellArray> newPolylines = vtkSmartPointer<vtkCellArray>::New();

	vtkSmartPointer<vtkPoints> points = inputPolyData->GetPoints();
	vtkSmartPointer<vtkCellArray> lines = inputPolyData->GetLines();

	vtkSmartPointer<vtkIdList> pointUsageCount = vtkSmartPointer<vtkIdList>::New();
	pointUsageCount->SetNumberOfIds(points->GetNumberOfPoints());
	pointUsageCount->Fill(0);

	// Count how many times each point is used in the polyline
	lines->InitTraversal();
	vtkSmartPointer<vtkIdList> pts = vtkSmartPointer<vtkIdList>::New();
	vtkIdType npts;
	while (lines->GetNextCell(pts)) {
		npts = pts->GetNumberOfIds();
		for (vtkIdType i = 0; i < npts; i++) {
			pointUsageCount->SetId(pts->GetId(i), pointUsageCount->GetId(pts->GetId(i)) + 1);
		}
	}

	// Traverse and split the polylines at shared points
	lines->InitTraversal();
	while (lines->GetNextCell(pts)) {
		npts = pts->GetNumberOfIds();
		vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
		vtkSmartPointer<vtkIdList> newPolyLinePts = vtkSmartPointer<vtkIdList>::New();

		for (vtkIdType i = 0; i < npts; i++) {
			newPolyLinePts->InsertNextId(pts->GetId(i));

			// Check if this is a break point (used in multiple branches)
			if (pointUsageCount->GetId(pts->GetId(i)) > 1 && i != 0 && i != npts - 1) {
				polyLine->GetPointIds()->DeepCopy(newPolyLinePts);
				newPolylines->InsertNextCell(polyLine);
				polyLine = vtkSmartPointer<vtkPolyLine>::New();
				newPolyLinePts = vtkSmartPointer<vtkIdList>::New();
				newPolyLinePts->InsertNextId(pts->GetId(i));
			}
		}

		if (newPolyLinePts->GetNumberOfIds() > 1) {
			polyLine->GetPointIds()->DeepCopy(newPolyLinePts);
			newPolylines->InsertNextCell(polyLine);
		}
	}

	return newPolylines;
}

vtkSmartPointer<vtkPolyData> CreatePolyDataWithSeparatedBranches(vtkPolyData* inputPolyData) {
	// Step 1: Use the SeparatePolyLineBranches function to get the separated polylines
	vtkSmartPointer<vtkCellArray> separatedLines = SeparatePolyLineBranches(inputPolyData);

	// Step 2: Create a new vtkPolyData object
	vtkSmartPointer<vtkPolyData> newPolyData = vtkSmartPointer<vtkPolyData>::New();

	// Step 3: Set the points from the original polydata
	newPolyData->SetPoints(inputPolyData->GetPoints());

	// Step 4: Set the new separated lines into the new vtkPolyData
	newPolyData->SetLines(separatedLines);

	return newPolyData;
}

float getSkeletonLength(vtkDataSet* skeleton)
{
	float length = 0.0;
	vtkSmartPointer<vtkIdList> cellPtIds = vtkSmartPointer<vtkIdList>::New();
	double pt1[3], pt2[3];
	for (int i = 0; i < skeleton->GetNumberOfCells(); i++)
	{
		skeleton->GetCellPoints(i, cellPtIds);
		skeleton->GetPoint(cellPtIds->GetId(0), pt1);
		skeleton->GetPoint(cellPtIds->GetId(1), pt2);
		length += vtkMath::Distance2BetweenPoints(pt1, pt2);
	}
	return length;
}

void ExtractRegions(vtkUnstructuredGrid* inputGrid, int& numRegions,
	std::vector<vtkSmartPointer<vtkUnstructuredGrid>>& regions)
{
	vtkDataArray* regionArray = inputGrid->GetCellData()->GetArray("RegionId");

	// Step 2: Prepare maps for each region
	std::vector<std::map<vtkIdType, vtkIdType>> globalToLocalPointMap(numRegions);
	std::vector<vtkSmartPointer<vtkPoints>> regionPoints(numRegions);
	std::vector<vtkSmartPointer<vtkCellArray>> regionCells(numRegions);

	for (int r = 0; r < numRegions; ++r) {
		regionPoints[r] = vtkSmartPointer<vtkPoints>::New();
		regionCells[r] = vtkSmartPointer<vtkCellArray>::New();
	}

	// Step 3: Allocate new point data arrays (copied from input)
	std::vector<vtkSmartPointer<vtkPointData>> regionPointData(numRegions);
	std::vector<vtkSmartPointer<vtkCellData>> regionCellData(numRegions);

	for (int r = 0; r < numRegions; ++r) {
		regionPointData[r] = vtkSmartPointer<vtkPointData>::New();
		regionPointData[r]->CopyAllocate(inputGrid->GetPointData());

		regionCellData[r] = vtkSmartPointer<vtkCellData>::New();
		regionCellData[r]->CopyAllocate(inputGrid->GetCellData());
	}

	// Step 4: Process each cell and assign to region
	vtkIdList* pointIds = vtkIdList::New();

	for (vtkIdType cellId = 0; cellId < inputGrid->GetNumberOfCells(); ++cellId) {
		int regionId = regionArray->GetTuple1(cellId);
		inputGrid->GetCellPoints(cellId, pointIds);

		std::vector<vtkIdType> localPtIds(pointIds->GetNumberOfIds());

		for (vtkIdType j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			vtkIdType globalId = pointIds->GetId(j);

			// Check if already inserted
			if (globalToLocalPointMap[regionId].find(globalId) == globalToLocalPointMap[regionId].end()) {
				vtkIdType newId = regionPoints[regionId]->InsertNextPoint(inputGrid->GetPoint(globalId));
				globalToLocalPointMap[regionId][globalId] = newId;

				// Copy point data
				regionPointData[regionId]->CopyData(inputGrid->GetPointData(), globalId, newId);
			}

			localPtIds[j] = globalToLocalPointMap[regionId][globalId];
		}

		regionCells[regionId]->InsertNextCell(static_cast<vtkIdType>(localPtIds.size()), localPtIds.data());

		// Copy cell data
		regionCellData[regionId]->CopyData(inputGrid->GetCellData(), cellId, regionCells[regionId]->GetNumberOfCells() - 1);
	}

	// Step 5: Assemble vtkUnstructuredGrid for each region
	for (int r = 0; r < numRegions; ++r) {
		regions[r] = vtkSmartPointer<vtkUnstructuredGrid>::New();
		regions[r]->SetPoints(regionPoints[r]);
		regions[r]->SetCells(VTK_LINE, regionCells[r]);  // Replace with actual cell type
		regionPointData[r]->Squeeze();
		regionCellData[r]->Squeeze();
		regions[r]->GetPointData()->DeepCopy(regionPointData[r]);
		regions[r]->GetCellData()->DeepCopy(regionCellData[r]);
	}
}

float checkVortexConsistency(vtkPolyData* skeleton) {
	double pt1[3], pt2[3], currVec[3], vortVec[3];
	vtkDataArray* vorticityArray = skeleton->GetPointData()->GetArray("vorticity");
	// cout << "Checking Vorticity Consistency" << endl;
	// We first find which end of the skeleton is the right most end.
	skeleton->GetPoint(0, pt1);
	skeleton->GetPoint(skeleton->GetNumberOfPoints() - 1, pt2);
	int startIdx;
	if (pt1[1] > pt2[1]) {	// Point count starts from the left end
		startIdx = skeleton->GetNumberOfPoints() - 1;
	}
	else {
		startIdx = 0;
	}

	int countPos = 0, countNeg = 0;
	for (int i = 0; i < skeleton->GetNumberOfPoints() - 1; i++) {
		skeleton->GetPoint(abs(i - startIdx), pt1);
		skeleton->GetPoint(abs(i - startIdx + 1), pt2);
		vorticityArray->GetTuple(abs(i - startIdx), vortVec);
		currVec[0] = pt2[0] - pt1[0];
		currVec[1] = pt2[1] - pt1[1];
		currVec[2] = pt2[2] - pt1[2];
		vtkMath::Normalize(vortVec);
		vtkMath::Normalize(currVec);
		float dot = vtkMath::Dot(vortVec, currVec);
		if (dot > 0.5) {
			countPos++;
		}
		else if (dot < -0.5) {
			countNeg++;
		}
		// cout << dot << " ";
	}
	// cout << endl;
	float numCells = skeleton->GetNumberOfCells();
	return std::max((float)countPos / numCells, (float)countNeg / numCells);
}

double getHausDist(vtkPolyData* candidateSkel, vtkPolyData* streamlines) {

	vtkSmartPointer<vtkHausdorffDistancePointSetFilter> distance =
		vtkSmartPointer<vtkHausdorffDistancePointSetFilter>::New();
	distance->SetInputData(0, candidateSkel);
	distance->SetInputData(1, streamlines);
	distance->Update();

	double distanceBeforeAlign = static_cast<vtkPointSet*>(distance->GetOutput(0))
		->GetFieldData()
		->GetArray("HausdorffDistance")
		->GetComponent(0, 0);

	return distanceBeforeAlign;
}

bool checkForHairpinType(vtkPolyData* skeleton, int& type)
{
	float totalLen = getSkeletonLength(skeleton);
	float lenUpTh = totalLen * 0.0;
	float lenBotTh = totalLen * 0.0;

	// Get spanwise section
	vtkSmartPointer<vtkThreshold> threshold1 = vtkSmartPointer<vtkThreshold>::New();
	threshold1->SetInputData(skeleton);
	threshold1->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "cellDir");
	threshold1->SetLowerThreshold(1);
	threshold1->SetUpperThreshold(1);
	threshold1->Update();

	// Get the largest segment of the spanwise section
	vtkSmartPointer<vtkConnectivityFilter> connFilter1 = vtkSmartPointer<vtkConnectivityFilter>::New();
	connFilter1->SetInputData(threshold1->GetOutput());
	connFilter1->SetExtractionModeToAllRegions();
	connFilter1->ColorRegionsOn();
	connFilter1->Update();

	int numRegs = connFilter1->GetNumberOfExtractedRegions();
	std::vector<vtkSmartPointer<vtkUnstructuredGrid>> regions1(numRegs);
	ExtractRegions(vtkUnstructuredGrid::SafeDownCast(connFilter1->GetOutput()), numRegs, regions1);

	vtkSmartPointer<vtkUnstructuredGrid> spanwiseSegment = vtkSmartPointer<vtkUnstructuredGrid>::New();
	double spanwiseZMax = skeleton->GetBounds()[4];
	for (vtkSmartPointer<vtkUnstructuredGrid> region : regions1) {

		if (getSkeletonLength(region) < lenBotTh)
		{
			continue;
		}

		if (spanwiseZMax < region->GetBounds()[5]) {
			spanwiseSegment->DeepCopy(region);
			spanwiseZMax = region->GetBounds()[5];
		}

	}
	regions1.clear();

	if (spanwiseSegment->GetNumberOfCells() == 0)
	{
		// cout << "No spanwise segment found." << endl;
		return false;
	}
	// cout << "Head found!" << endl;

	vtkDataArray* spOyfs = spanwiseSegment->GetPointData()->GetArray("oyf");
	float spOyfAvg = 0.0;
	for (int i = 0; i < spOyfs->GetNumberOfTuples(); i++) {
		spOyfAvg += spOyfs->GetTuple1(i);
	}
	spOyfAvg /= spOyfs->GetNumberOfTuples();
	if (spOyfAvg < 0.0)
	{
		// cout << "Oyf check failed!" << endl;
		return false;
	}

	vtkDataArray* segmentCellIdsArray = spanwiseSegment->GetCellData()->GetArray("cellIds");
	vtkIdType spanwiseMinCellId = skeleton->GetNumberOfCells();
	vtkIdType spanwiseMaxCellId = -1;
	for (int i = 0; i < segmentCellIdsArray->GetNumberOfTuples(); i++) {
		vtkIdType cellId = segmentCellIdsArray->GetTuple1(i);
		if (cellId > spanwiseMaxCellId) {
			spanwiseMaxCellId = cellId;
		}
		if (cellId < spanwiseMinCellId) {
			spanwiseMinCellId = cellId;
		}
	}
	double spBounds[6];
	spanwiseSegment->GetBounds(spBounds);

	// Get streamwise section
	vtkSmartPointer<vtkThreshold> threshold2 = vtkSmartPointer<vtkThreshold>::New();
	threshold2->SetInputData(skeleton);
	threshold2->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "cellDir");
	threshold2->SetLowerThreshold(0);
	threshold2->SetUpperThreshold(0);
	threshold2->Update();

	// Get all the disconnected segments of the streamwise section
	vtkSmartPointer<vtkConnectivityFilter> connFilter2 = vtkSmartPointer<vtkConnectivityFilter>::New();
	connFilter2->SetInputData(threshold2->GetOutput());
	connFilter2->SetExtractionModeToAllRegions();
	connFilter2->ColorRegionsOn();
	connFilter2->Update();

	numRegs = connFilter2->GetNumberOfExtractedRegions();
	std::vector<vtkSmartPointer<vtkUnstructuredGrid>> regions2(numRegs);
	ExtractRegions(vtkUnstructuredGrid::SafeDownCast(connFilter2->GetOutput()), numRegs, regions2);

	// Check how many streamwise sections are there? 
	// If there are more than two get the largest two otherwise don't. 
	// This is because a hairpin vortex could have maximum two legs.
	vtkSmartPointer<vtkUnstructuredGrid> leftStreamwiseSection = vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtkSmartPointer<vtkUnstructuredGrid> rightStreamwiseSection = vtkSmartPointer<vtkUnstructuredGrid>::New();
	float longestLeftSide = 0.0, longestRightSide = 0.0;
	// int leftMinCellId = -1, rightMaxCellId = segmentCellIdsArray->GetNumberOfTuples();
	for (vtkSmartPointer<vtkUnstructuredGrid> region : regions2)
	{
		float legLen = getSkeletonLength(region);

		if (legLen < lenBotTh)
		{
			continue;
		}

		vtkDataArray* segmentCellIdsArray = region->GetCellData()->GetArray("cellIds");
		vtkIdType minCellId = skeleton->GetNumberOfCells();
		vtkIdType maxCellId = -1;
		for (int i = 0; i < segmentCellIdsArray->GetNumberOfTuples(); i++) {
			vtkIdType cellId = segmentCellIdsArray->GetTuple1(i);
			if (cellId > maxCellId) {
				maxCellId = cellId;
			}
			if (cellId < minCellId) {
				minCellId = cellId;
			}
		}

		if (minCellId > spanwiseMaxCellId && longestLeftSide < legLen)
		{
			// leftMinCellId = minCellId;
			longestLeftSide = legLen;
			leftStreamwiseSection->DeepCopy(region);
		}

		if (maxCellId < spanwiseMinCellId && longestRightSide < legLen)
		{
			// rightMaxCellId = maxCellId;
			longestRightSide = legLen;
			rightStreamwiseSection->DeepCopy(region);
		}
	}
	regions2.clear();

	int numLegs;
	int behindLegs;
	int belowLegs;
	float legLen;
	vtkSmartPointer<vtkUnstructuredGrid> legSection = nullptr;
	double legBounds[6];
	double st1Bounds[6], st2Bounds[6];
	if (longestLeftSide > 0.0 && longestRightSide > 0.0) // There are two legs
	{
		// cout << "Two legs found!" << endl;
		numLegs = 2;
		leftStreamwiseSection->GetBounds(st1Bounds);
		rightStreamwiseSection->GetBounds(st2Bounds);

		if (st1Bounds[3] < st2Bounds[3])
		{
			// cout << "Swap left and right leg" << endl;
			vtkSmartPointer<vtkUnstructuredGrid> temp = leftStreamwiseSection;
			leftStreamwiseSection = rightStreamwiseSection;
			rightStreamwiseSection = temp;
		}
		leftStreamwiseSection->GetBounds(st1Bounds);
		rightStreamwiseSection->GetBounds(st2Bounds);
		float leftLegLen = getSkeletonLength(leftStreamwiseSection);
		float rightLegLen = getSkeletonLength(rightStreamwiseSection);

		// In case one of the leg is above of the head, then it should be very small.
		// It could happen because of the skeleton extraction method.
		if (spBounds[5] - st1Bounds[5] > 0 && spBounds[5] - st2Bounds[5] > 0) {
			// cout << "Both legs are below the head!" << endl;
			belowLegs = 2;
		}
		else if (spBounds[5] - st1Bounds[5] < 0 && spBounds[5] - st2Bounds[5] > 0) {
			if (leftLegLen < lenUpTh)
			{
				// cout << "Left leg is above the head!" << endl;
				if (rightLegLen < lenBotTh) {
					// cout << "Right leg len is too short!" << endl;
					return false;
				}
				legSection = rightStreamwiseSection;
				legLen = rightLegLen;
				legSection->GetBounds(legBounds);
				belowLegs = 1;
			}
			else
			{
				// cout << "Left leg is above the head and it is too long!" << endl;
				belowLegs = 0;
				return false;
			}
		}
		else if (spBounds[5] - st1Bounds[5] > 0 && spBounds[5] - st2Bounds[5] < 0) {
			if (rightLegLen < lenUpTh)
			{
				// cout << "Right leg is above the head!" << endl;
				if (leftLegLen < lenBotTh) {
					// cout << "Left leg len is too short!" << endl;
					return false;
				}
				legSection = leftStreamwiseSection;
				legLen = leftLegLen;
				legSection->GetBounds(legBounds);
				belowLegs = 1;
			}
			else
			{
				// cout << "Right leg is above the head and it is too long!" << endl;
				belowLegs = 0;
				return false;
			}
		}
		else {
			// cout << "Both legs are above the head!" << endl;
			return false;
		}

		// The x-max of the bounds of the spanwise segments should be further than those of streamwise segments
		if (spBounds[1] - st1Bounds[1] > 0 && spBounds[1] - st2Bounds[1] > 0)
		{
			// cout << "Both legs are behind the head!" << endl;
			behindLegs = 2;
		}
		else if (spBounds[1] - st1Bounds[1] < 0 && spBounds[1] - st2Bounds[1] > 0)
		{
			// In case one of the leg is ahead of the head, then it should be very small.
			// It could happen because of the skeleton extraction method.
			if (leftLegLen < lenUpTh)
			{
				// cout << "Left leg is ahead of the head!" << endl;
				if (rightLegLen < lenBotTh) {
					// cout << "Right leg len is too short!" << endl;
					return false;
				}
				legSection = rightStreamwiseSection;
				legLen = rightLegLen;
				legSection->GetBounds(legBounds);
				behindLegs = 1;
			}
			else
			{
				// cout << "Left leg is ahead of the head but it is too long!" << endl;
				behindLegs = 0;
				return false;
			}
		}
		else if (spBounds[1] - st1Bounds[1] > 0 && spBounds[1] - st2Bounds[1] < 0)
		{
			// In case one of the leg is ahead of the head, then it should be very small.
			// It could happen because of the skeleton extraction method.
			if (rightLegLen < lenUpTh)
			{
				// cout << "Right leg is ahead of the head!" << endl;
				if (leftLegLen < lenBotTh) {
					// cout << "Left leg len is too short!" << endl;
					return false;
				}
				legSection = leftStreamwiseSection;
				legLen = leftLegLen;
				legSection->GetBounds(legBounds);
				behindLegs = 1;
			}
			else
			{
				// cout << "Right leg is ahead of the head but it is too long!" << endl;
				behindLegs = 0;
				return false;
			}
		}
		else
		{
			// cout << "Both legs are ahead of the head!" << endl;
			return false;
		}
		// cout << "Leg Streamwise and Elevation check passed!" << endl;

		// Now we find the average values of vor_x for both streamwise segments
		vtkDataArray* leftLegVorticity = leftStreamwiseSection->GetPointData()->GetArray("vorticity");
		vtkDataArray* rightLegVorticity = rightStreamwiseSection->GetPointData()->GetArray("vorticity");
		float leftLegVorticityXAvg = 0.0, rightLegVorticityXAvg = 0.0;
		for (int i = 0; i < leftLegVorticity->GetNumberOfTuples(); i++) {
			leftLegVorticityXAvg += leftLegVorticity->GetTuple(i)[0];
		}
		leftLegVorticityXAvg /= leftLegVorticity->GetNumberOfTuples();

		for (int i = 0; i < rightLegVorticity->GetNumberOfTuples(); i++) {
			rightLegVorticityXAvg += rightLegVorticity->GetTuple(i)[0];
		}
		rightLegVorticityXAvg /= rightLegVorticity->GetNumberOfTuples();
		if (behindLegs == 2) {

			if (leftLegVorticityXAvg > 0)
			{
				// cout << "Left leg scalar check failed!" << endl;
				return false;
			}

			if (rightLegVorticityXAvg < 0)
			{
				// cout << "Right leg scalar check failed!" << endl;
				return false;
			}
		}
		else if (behindLegs == 1) {
			// In case if one leg is ahead of the head, then both the legs should have the same sign
			if (leftLegVorticityXAvg * rightLegVorticityXAvg < 0) {
				// cout << "Both the legs should have the opposite sign" << endl;
				return false;
			}
		}
		// cout << "Leg Scalars check passed!" << endl;
	}
	else if (longestLeftSide > 0.0 || longestRightSide > 0.0) // There is one leg
	{
		// cout << "Only one leg is found!" << endl;
		numLegs = 1;
		if (longestLeftSide > 0.0)
		{
			legSection = leftStreamwiseSection;
		}
		else if (longestRightSide > 0.0)
		{
			legSection = rightStreamwiseSection;
		}
		legLen = getSkeletonLength(legSection);

		legSection->GetBounds(legBounds);

		if (spBounds[5] - legBounds[5] < 0)
		{
			// cout << "Leg is above the head!" << endl;
			return false;
		}

		if (spBounds[1] - legBounds[1] < 0)
		{
			// cout << "Leg is ahead of the head!" << endl;
			return false;
		}
		// cout << "Leg Streamwise and Elevation check passed!" << endl;

		vtkDataArray* legVorticity = legSection->GetPointData()->GetArray("vorticity");
		float legVorticityXAvg = 0.0;

		for (int i = 0; i < legVorticity->GetNumberOfTuples(); i++) {
			legVorticityXAvg += legVorticity->GetTuple(i)[0];
		}
		legVorticityXAvg /= legVorticity->GetNumberOfTuples();

		// Right leg can't have a negative VorticityXAvg and left leg vice versa.
		if (legVorticityXAvg > 0 && spBounds[3] - legBounds[3] < 0)
		{
			// cout << "Right Leg vorticity check failed" << endl;
			return false;
		}
		else if (legVorticityXAvg < 0 && spBounds[3] - legBounds[3] > 0)
		{
			// cout << "Left Leg vorticity check failed" << endl;
			return false;
		}
		// cout << "Leg Scalars check passed!" << endl;
	}
	else
	{
		// cout << "No legs are found!" << endl;
		numLegs = 0;
	}

	// Get neck section
	threshold2 = vtkSmartPointer<vtkThreshold>::New();
	threshold2->SetInputData(skeleton);
	threshold2->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "cellDir");
	threshold2->SetLowerThreshold(2);
	threshold2->SetUpperThreshold(2);
	threshold2->Update();

	// Get all the disconnected segments of the neck section
	connFilter2 = vtkSmartPointer<vtkConnectivityFilter>::New();
	connFilter2->SetInputData(threshold2->GetOutput());
	connFilter2->SetExtractionModeToAllRegions();
	connFilter2->ColorRegionsOn();
	connFilter2->Update();

	numRegs = connFilter2->GetNumberOfExtractedRegions();
	std::vector<vtkSmartPointer<vtkUnstructuredGrid>> regions3(numRegs);
	ExtractRegions(vtkUnstructuredGrid::SafeDownCast(connFilter2->GetOutput()), numRegs, regions3);

	// Check how many neck sections are there? 
	// If there are more than two get the largest two otherwise don't. 
	// This is because a hairpin vortex could have maximum two necks.
	vtkSmartPointer<vtkUnstructuredGrid> leftNeckSection = vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtkSmartPointer<vtkUnstructuredGrid> rightNeckSection = vtkSmartPointer<vtkUnstructuredGrid>::New();
	float leftNeckSide = 0.0, rightNeckSide = 0.0;
	// int leftNeckCellId = -1, rightNeckCellId = segmentCellIdsArray->GetNumberOfTuples();
	for (vtkSmartPointer<vtkUnstructuredGrid> region : regions3)
	{
		float neckLen = getSkeletonLength(region);
		if (neckLen < lenBotTh)
		{
			continue;
		}

		vtkDataArray* segmentCellIdsArray = region->GetCellData()->GetArray("cellIds");
		vtkIdType minCellId = skeleton->GetNumberOfCells();
		vtkIdType maxCellId = -1;
		for (int i = 0; i < segmentCellIdsArray->GetNumberOfTuples(); i++) {
			vtkIdType cellId = segmentCellIdsArray->GetTuple1(i);
			if (cellId > maxCellId) {
				maxCellId = cellId;
			}
			if (cellId < minCellId) {
				minCellId = cellId;
			}
		}

		if (minCellId > spanwiseMaxCellId && leftNeckSide < neckLen)
		{
			// leftNeckCellId = minCellId;
			leftNeckSide = neckLen;
			leftNeckSection->DeepCopy(region);
		}

		if (maxCellId < spanwiseMinCellId && rightNeckSide < neckLen)
		{
			// rightNeckCellId = maxCellId;
			rightNeckSide = neckLen;
			rightNeckSection->DeepCopy(region);
		}
	}
	regions3.clear();

	if (leftNeckSide > 0.0 && rightNeckSide > 0.0) // There are two necks
	{
		// cout << "Two necks are found!" << endl;
		double neck1Bounds[6], neck2Bounds[6];
		leftNeckSection->GetBounds(neck1Bounds);
		rightNeckSection->GetBounds(neck2Bounds);

		if (neck1Bounds[3] < neck2Bounds[3])
		{
			// cout << "Swap left and right neck" << endl;
			vtkSmartPointer<vtkUnstructuredGrid> temp = leftNeckSection;
			leftNeckSection = rightNeckSection;
			rightNeckSection = temp;
		}
		leftNeckSection->GetBounds(neck1Bounds);
		rightNeckSection->GetBounds(neck2Bounds);

		if (spBounds[5] - neck1Bounds[5] < 0)
		{
			// cout << "Left neck is above the head!" << endl;
			return false;
		}

		if (spBounds[5] - neck2Bounds[5] < 0)
		{
			// cout << "Right neck is above the head!" << endl;
			return false;
		}
		// cout << "Neck Elevation check passed!" << endl;

		vtkDataArray* leftNeckVorticity = leftNeckSection->GetPointData()->GetArray("vorticity");
		vtkDataArray* rightNeckVorticity = rightNeckSection->GetPointData()->GetArray("vorticity");
		float leftNeckVorticityZAvg = 0.0, rightNeckVorticityZAvg = 0.0;
		for (int i = 0; i < leftNeckVorticity->GetNumberOfTuples(); i++) {
			leftNeckVorticityZAvg += leftNeckVorticity->GetTuple(i)[2];
		}
		leftNeckVorticityZAvg /= leftNeckVorticity->GetNumberOfTuples();

		for (int i = 0; i < rightNeckVorticity->GetNumberOfTuples(); i++) {
			rightNeckVorticityZAvg += rightNeckVorticity->GetTuple(i)[2];
		}
		rightNeckVorticityZAvg /= rightNeckVorticity->GetNumberOfTuples();

		if (leftNeckVorticityZAvg > 0)
		{
			// cout << "Left neck vorticity check failed!" << endl;
			return false;
		}
		if (rightNeckVorticityZAvg < 0)
		{
			// cout << "Right neck vorticity check failed!" << endl;
			return false;
		}
		// cout << "Neck Scalars check passed!" << endl;

		if (numLegs == 2 && behindLegs == 2 && belowLegs == 2)
		{
			// cout << "There are two legs and two necks and both the legs are below and behind the head!" << endl;
			type = 1;
		}
		else if (numLegs == 1)
		{
			// cout << "There is one leg and two necks and the leg is behind the head!" << endl;
			// We need to check, which side the leg is on
			if (abs(neck1Bounds[3] - legBounds[3]) < abs(neck2Bounds[3] - legBounds[3])) {
				// cout << "The leg is on the left side" << endl;
				// Now we need to check if the leg is below and behind the left neck
				// If the leg is below the neck, then it should be behind as well
				if (neck1Bounds[5] - legBounds[5] > 0) {
					if (neck1Bounds[1] - legBounds[1] < 0) {
						// cout << "Leg is ahead of the left neck" << endl;
						return false;
					}
					// cout << "Leg is below and behind the left neck." << endl;
				}
			}
			else {
				// cout << "The leg is on the right side" << endl;
				// Now we need to check if the leg is below and behind the right neck
				// If the leg is below the neck, then it should be behind as well
				if (neck2Bounds[5] - legBounds[5] > 0) {
					if (neck2Bounds[1] - legBounds[1] < 0) {
						// cout << "Leg is ahead of the right neck" << endl;
						return false;
					}
					// cout << "Leg is below and behind the right neck." << endl;
				}
			}
			type = 4;
		}
		else if (numLegs == 0)
		{
			// cout << "There is no leg but two necks!" << endl;
			type = 5;
		}
	}
	else if (leftNeckSide > 0.0 || rightNeckSide > 0.0) // There is one neck
	{
		// cout << "There is one neck!" << endl;
		vtkSmartPointer<vtkUnstructuredGrid> neckSection = nullptr;
		if (leftNeckSide > 0.0)
		{
			neckSection = leftNeckSection;
		}
		else if (rightNeckSide > 0.0)
		{
			neckSection = rightNeckSection;
		}

		double neckBounds[6];
		neckSection->GetBounds(neckBounds);

		if (spBounds[5] - neckBounds[5] < 0)
		{
			// cout << "The neck is above the head!" << endl;
			return false;
		}
		// cout << "Neck Elevation check passed!" << endl;

		vtkDataArray* neckVorticity = neckSection->GetPointData()->GetArray("vorticity");
		float neckVorticityZAvg = 0.0;

		for (int i = 0; i < neckVorticity->GetNumberOfTuples(); i++) {
			neckVorticityZAvg += neckVorticity->GetTuple(i)[2];
		}
		neckVorticityZAvg /= neckVorticity->GetNumberOfTuples();

		// Right neck can't have a negative VorticityZAvg and left neck vice versa.
		if ((neckVorticityZAvg > 0 && spBounds[3] - neckBounds[3] < 0))
		{
			// cout << "Right neck vorticity check failed" << endl;
			return false;
		}

		if ((neckVorticityZAvg < 0 && spBounds[3] - neckBounds[3] > 0))
		{
			// cout << "Left neck vorticity check failed" << endl;
			return false;
		}

		// cout << "Neck Scalars check passed!" << endl;
		if (numLegs == 2 && behindLegs == 2 && belowLegs == 2)
		{
			// cout << "There are two legs and one neck and both of the legs are below and behind the head!" << endl;
			// We need to check, which side the neck is on
			if (abs(st1Bounds[3] - neckBounds[3]) < abs(st2Bounds[3] - neckBounds[3])) {
				// cout << "The neck is on the left side" << endl;
				// Now we need to check if the left leg is below and behind the neck
				// If the left leg is below the neck, then it should be behind as well
				if (neckBounds[5] - st1Bounds[5] > 0) {
					if (neckBounds[1] - st1Bounds[1] < 0) {
						// cout << "Left Leg is ahead of the neck" << endl;
						return false;
					}
					// cout << "Leg is below and behind the left neck." << endl;
				}
			}
			else {
				// cout << "The neck is on the right side" << endl;
				// Now we need to check if the right leg is below and behind the neck
				// If the right leg is below the neck, then it should be behind as well
				if (neckBounds[5] - st2Bounds[5] > 0) {
					if (neckBounds[1] - st2Bounds[1] < 0) {
						// cout << "Right Leg is ahead of the neck" << endl;
						return false;
					}
					// cout << "Leg is below and behind the right neck." << endl;
				}
			}
			type = 2;
		}
		else
		{
			// cout << "There is no leg and one neck!" << endl;
			return false;
		}
	}
	else // There is no neck
	{
		// cout << "There is no neck!" << endl;
		return false;
	}

	return true;
}

void getSubSegments(std::set<vtkIdType>& uniqueSegmentIds, std::set<std::set<vtkIdType>>& subSegments)
{
	// We check all the sets in the power set of uniqueRegionIds and only shortlist those which has size >= minSetSize
	int minSetSize = 1;
	// cout << minSetSize << endl;
	int powSetSize = pow(2, uniqueSegmentIds.size());
	int i, j;
	for (i = 0; i < powSetSize; i++) {
		std::set<vtkIdType> tempSet;
		for (j = 0; j < uniqueSegmentIds.size(); j++) {
			// Check if jth bit in the counter is set, If set then add the jth element from uniqueRegionIds 
			if (i & (1 << j)) {
				auto it = std::next(uniqueSegmentIds.begin(), j);
				tempSet.insert(*it);
			}
		}
		if (tempSet.size() > minSetSize)
		{
			subSegments.insert(tempSet);
		}
	}
}

bool checkSegment(vtkPolyData* skel) {
	vtkSmartPointer<vtkIntArray> cellIdsArray = vtkSmartPointer<vtkIntArray>::New();
	cellIdsArray->SetNumberOfComponents(1);
	cellIdsArray->SetNumberOfTuples(skel->GetNumberOfCells());
	cellIdsArray->SetName("cellIds");
	for (int i = 0; i < skel->GetNumberOfCells(); i++) {
		cellIdsArray->SetTuple1(i, i);
	}
	skel->GetCellData()->AddArray(cellIdsArray);

	// Add a cell array to keep track of spanwise(1) and streamwise(0) segments.
	vtkSmartPointer<vtkUnsignedCharArray> cellDirArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
	cellDirArray->SetNumberOfComponents(1);
	cellDirArray->SetNumberOfTuples(skel->GetNumberOfCells());
	cellDirArray->SetName("cellDir");
	vtkSmartPointer<vtkIdList> cellPts = vtkSmartPointer<vtkIdList>::New();
	double pt1[3], pt2[3], currVec[3];
	double i_unit[3] = { 1.0, 0.0, 0.0 };
	double j_unit[3] = { 0.0, 1.0, 0.0 };
	double k_unit[3] = { 0.0, 0.0, 1.0 };
	float xDir, yDir, zDir;
	int numY = 0, numZ = 0;
	for (vtkIdType cellId = 0; cellId < skel->GetNumberOfCells(); cellId++) {
		skel->GetCellPoints(cellId, cellPts);
		skel->GetPoint(cellPts->GetId(0), pt1);
		skel->GetPoint(cellPts->GetId(1), pt2);

		currVec[0] = pt2[0] - pt1[0]; currVec[1] = pt2[1] - pt1[1]; currVec[2] = pt2[2] - pt1[2];

		vtkMath::Normalize(currVec);

		xDir = vtkMath::Dot(currVec, i_unit);
		xDir = std::sqrt(xDir * xDir);	// Get absolute value
		yDir = vtkMath::Dot(currVec, j_unit);
		yDir = std::sqrt(yDir * yDir);	// Get absolute value
		zDir = vtkMath::Dot(currVec, k_unit);
		zDir = std::sqrt(zDir * zDir);	// Get absolute value
		// cout << xDir << " " << yDir << " " << zDir << endl;

		if (yDir > xDir && yDir > zDir && yDir > 0.5)
		{
			cellDirArray->SetTuple1(cellId, 1);
			numY++;
		}
		else if (zDir > xDir && zDir > yDir && zDir > 0.5)
		{
			cellDirArray->SetTuple1(cellId, 2);
			numZ++;
		}
		else if (xDir > yDir && xDir > zDir && xDir > 0.5)
		{
			cellDirArray->SetTuple1(cellId, 0);
		}
		else
		{
			cellDirArray->SetTuple1(cellId, 3);
		}
	}

	if (numY == 0 || numZ == 0) {
		return false;
	}
	skel->GetCellData()->AddArray(cellDirArray);
	skel->GetCellData()->SetActiveScalars(cellDirArray->GetName());
	return true;
}

bool checkStreamSegment(vtkPolyData* streamlines) {

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(streamlines);
	triangleFilter->Update();

	vtkSmartPointer<vtkConnectivityFilter> connFilter = vtkSmartPointer<vtkConnectivityFilter>::New();
	connFilter->SetInputData(triangleFilter->GetOutput());
	connFilter->SetRegionIdAssignmentMode(2);
	connFilter->SetExtractionModeToAllRegions();
	connFilter->ColorRegionsOn();
	connFilter->Update();

	for (int i = 0; i < connFilter->GetNumberOfExtractedRegions(); i++)
	{
		vtkSmartPointer<vtkThreshold> threshold = vtkSmartPointer<vtkThreshold>::New();
		threshold->SetInputData(connFilter->GetOutput());
		threshold->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "RegionId");
		threshold->SetLowerThreshold(i);
		threshold->SetUpperThreshold(i);
		threshold->Update();

		vtkSmartPointer<vtkGeometryFilter> surfaceFilter = vtkSmartPointer<vtkGeometryFilter>::New();
		surfaceFilter->SetInputData(threshold->GetOutput());
		surfaceFilter->Update();

		vtkPolyData* streamline = surfaceFilter->GetOutput();

		if (checkSegment(streamline)) {
			return true;
		}
	}

	return false;
}

void buildAdjacencyList(vtkPolyData* polyData,
	std::unordered_map<vtkIdType, std::vector<vtkIdType>>& adjacency) {
	for (vtkIdType i = 0; i < polyData->GetNumberOfCells(); ++i) {
		vtkCell* cell = polyData->GetCell(i);
		if (cell->GetCellType() == VTK_LINE) {
			vtkIdType p0 = cell->GetPointId(0);
			vtkIdType p1 = cell->GetPointId(1);
			adjacency[p0].push_back(p1);
			adjacency[p1].push_back(p0);  // undirected
		}
	}
}

void dfsLongest(vtkIdType current, vtkIdType end,
	const std::unordered_map<vtkIdType, std::vector<vtkIdType>>& graph,
	std::unordered_set<vtkIdType>& visited,
	std::vector<vtkIdType>& currentPath,
	std::vector<vtkIdType>& longestPath) {
	visited.insert(current);
	currentPath.push_back(current);

	if (current == end) {
		if (currentPath.size() > longestPath.size())
			longestPath = currentPath;
	}
	else {
		for (vtkIdType neighbor : graph.at(current)) {
			if (visited.find(neighbor) == visited.end()) {
				dfsLongest(neighbor, end, graph, visited, currentPath, longestPath);
			}
		}
	}

	currentPath.pop_back();
	visited.erase(current);
}

std::vector<vtkIdType> bfsShortestPath(
	vtkIdType start,
	vtkIdType end,
	const std::unordered_map<vtkIdType, std::vector<vtkIdType>>& graph)
{
	std::unordered_map<vtkIdType, vtkIdType> cameFrom; // child -> parent
	std::unordered_set<vtkIdType> visited;
	std::queue<vtkIdType> q;

	q.push(start);
	visited.insert(start);

	bool found = false;

	while (!q.empty())
	{
		vtkIdType current = q.front();
		q.pop();

		if (current == end)
		{
			found = true;
			break;
		}

		auto it = graph.find(current);
		if (it != graph.end())
		{
			for (vtkIdType neighbor : it->second)
			{
				if (visited.find(neighbor) == visited.end())
				{
					visited.insert(neighbor);
					cameFrom[neighbor] = current;
					q.push(neighbor);
				}
			}
		}
	}

	std::vector<vtkIdType> path;
	if (found)
	{
		// Reconstruct path backwards
		for (vtkIdType node = end; node != start; node = cameFrom[node])
		{
			path.push_back(node);
		}
		path.push_back(start);
		std::reverse(path.begin(), path.end());
	}

	return path; // empty if no path found
}

void dfsAllPaths(
	vtkIdType current,
	vtkIdType end,
	const std::unordered_map<vtkIdType, std::vector<vtkIdType>>& graph,
	std::unordered_set<vtkIdType>& visited,
	std::vector<vtkIdType>& currentPath,
	std::vector<std::vector<vtkIdType>>& allPaths)
{
	visited.insert(current);
	currentPath.push_back(current);

	if (current == end) {
		allPaths.push_back(currentPath);
	}
	else {
		auto it = graph.find(current);
		if (it != graph.end()) {
			for (vtkIdType neighbor : it->second) {
				if (visited.find(neighbor) == visited.end()) {
					dfsAllPaths(neighbor, end, graph, visited, currentPath, allPaths);
				}
			}
		}
	}

	currentPath.pop_back();
	visited.erase(current);  // allow reuse in other paths
}

void buildLinesFromPath(
	const std::vector<vtkIdType>& path,
	vtkPolyData* inputPoints,
	vtkSmartPointer<vtkPolyData> polyData)
{
	auto newPoints = vtkSmartPointer<vtkPoints>::New();
	auto lines = vtkSmartPointer<vtkCellArray>::New();
	auto pointData = vtkSmartPointer<vtkPointData>::New();
	auto cellData = vtkSmartPointer<vtkCellData>::New();

	// Prepare point data arrays
	vtkPointData* inputPointData = inputPoints->GetPointData();
	int numPointArrays = inputPointData->GetNumberOfArrays();
	std::vector<vtkSmartPointer<vtkDataArray>> copiedPointArrays(numPointArrays);

	for (int i = 0; i < numPointArrays; ++i)
	{
		vtkDataArray* inputArray = inputPointData->GetArray(i);
		if (inputArray)
		{
			auto newArray = vtkSmartPointer<vtkDataArray>::Take(inputArray->NewInstance());
			newArray->SetName(inputArray->GetName());
			newArray->SetNumberOfComponents(inputArray->GetNumberOfComponents());
			newArray->SetNumberOfTuples(path.size());
			copiedPointArrays[i] = newArray;
			pointData->AddArray(newArray);
		}
	}

	// Prepare cell data arrays
	vtkCellData* inputCellData = inputPoints->GetCellData();
	int numCellArrays = inputCellData->GetNumberOfArrays();
	std::vector<vtkSmartPointer<vtkDataArray>> copiedCellArrays(numCellArrays);

	vtkIdType numSegments = static_cast<vtkIdType>(path.size()) - 1;
	for (int i = 0; i < numCellArrays; ++i)
	{
		vtkDataArray* inputArray = inputCellData->GetArray(i);
		if (inputArray)
		{
			auto newArray = vtkSmartPointer<vtkDataArray>::Take(inputArray->NewInstance());
			newArray->SetName(inputArray->GetName());
			newArray->SetNumberOfComponents(inputArray->GetNumberOfComponents());
			newArray->SetNumberOfTuples(numSegments);
			copiedCellArrays[i] = newArray;
			cellData->AddArray(newArray);
		}
	}

	// Build points and point data
	for (vtkIdType i = 0; i < static_cast<vtkIdType>(path.size()); ++i)
	{
		double pt[3];
		vtkIdType originalId = path[i];
		inputPoints->GetPoint(originalId, pt);
		newPoints->InsertNextPoint(pt);

		// Copy point data
		for (int j = 0; j < numPointArrays; ++j)
		{
			if (copiedPointArrays[j])
			{
				copiedPointArrays[j]->SetTuple(i, inputPointData->GetArray(j)->GetTuple(originalId));
			}
		}
	}

	// For each segment, create a vtkLine cell and copy the corresponding cell data
	for (vtkIdType i = 0; i < numSegments; ++i)
	{
		vtkIdType origId0 = path[i];
		vtkIdType origId1 = path[i + 1];

		// Create line cell
		auto line = vtkSmartPointer<vtkLine>::New();
		line->GetPointIds()->SetId(0, i);     // local index in newPoints
		line->GetPointIds()->SetId(1, i + 1);
		lines->InsertNextCell(line);

		// Find the input cell connecting these two points
		vtkSmartPointer<vtkIdList> cellIds0 = vtkSmartPointer<vtkIdList>::New();
		vtkSmartPointer<vtkIdList> cellIds1 = vtkSmartPointer<vtkIdList>::New();
		inputPoints->GetPointCells(origId0, cellIds0);
		inputPoints->GetPointCells(origId1, cellIds1);

		vtkIdType sourceCellId = -1;
		for (vtkIdType c0 = 0; c0 < cellIds0->GetNumberOfIds(); ++c0)
		{
			vtkIdType cid = cellIds0->GetId(c0);
			// Check if this cell also contains origId1
			auto cellPts = vtkSmartPointer<vtkIdList>::New();
			inputPoints->GetCellPoints(cid, cellPts);
			bool has0 = false, has1 = false;
			for (vtkIdType k = 0; k < cellPts->GetNumberOfIds(); ++k)
			{
				vtkIdType pid = cellPts->GetId(k);
				if (pid == origId0) has0 = true;
				if (pid == origId1) has1 = true;
			}
			if (has0 && has1)
			{
				sourceCellId = cid;
				break;
			}
		}

		// Copy cell data from the source cell if found
		if (sourceCellId >= 0)
		{
			for (int j = 0; j < numCellArrays; ++j)
			{
				if (copiedCellArrays[j])
				{
					copiedCellArrays[j]->SetTuple(i, inputCellData->GetArray(j)->GetTuple(sourceCellId));
				}
			}
		}
	}

	// Set geometry and data
	polyData->SetPoints(newPoints);
	polyData->SetLines(lines);
	polyData->GetPointData()->ShallowCopy(pointData);
	polyData->GetCellData()->ShallowCopy(cellData);
}

void AssignBranchIdsWithJunctionSplitting(vtkPolyData* input)
{
	vtkIdType numCells = input->GetNumberOfCells();

	// Map: pointId -> list of connected cellIds
	std::unordered_map<vtkIdType, std::vector<vtkIdType>> pointToCells;
	for (vtkIdType cellId = 0; cellId < numCells; ++cellId)
	{
		auto idList = vtkSmartPointer<vtkIdList>::New();
		input->GetCellPoints(cellId, idList);
		for (vtkIdType i = 0; i < idList->GetNumberOfIds(); ++i)
		{
			vtkIdType pid = idList->GetId(i);
			pointToCells[pid].push_back(cellId);
		}
	}

	// Identify junction points
	std::unordered_set<vtkIdType> junctionPoints;
	for (const auto& kv : pointToCells)
	{
		if (kv.second.size() > 2)
		{
			junctionPoints.insert(kv.first);
		}
	}

	// Mark cells as visited
	std::vector<bool> visited(numCells, false);

	// Prepare output array
	auto branchIds = vtkSmartPointer<vtkIntArray>::New();
	branchIds->SetName("BranchIds");
	branchIds->SetNumberOfComponents(1);
	branchIds->SetNumberOfTuples(numCells);
	branchIds->FillComponent(0, -1); // Initialize with -1

	int currentBranchId = 0;

	for (vtkIdType cellId = 0; cellId < numCells; ++cellId)
	{
		if (visited[cellId])
			continue;

		auto idList = vtkSmartPointer<vtkIdList>::New();
		input->GetCellPoints(cellId, idList);

		// Determine if either endpoint is a junction
		vtkIdType p0 = idList->GetId(0);
		vtkIdType p1 = idList->GetId(1);

		// We will try to start traversal from both ends
		for (int end = 0; end < 2; ++end)
		{
			vtkIdType startPoint = (end == 0) ? p0 : p1;
			if (junctionPoints.count(startPoint))
				continue; // Can't start here, it's a junction

			// Traversal queue of cells
			std::queue<std::pair<vtkIdType, vtkIdType>> q;
			q.push({ cellId, startPoint });

			while (!q.empty())
			{
				auto front = q.front();
				vtkIdType currentCellId = front.first;
				vtkIdType fromPoint = front.second;
				q.pop();

				if (visited[currentCellId])
					continue;

				visited[currentCellId] = true;
				branchIds->SetValue(currentCellId, currentBranchId);

				auto currIdList = vtkSmartPointer<vtkIdList>::New();
				input->GetCellPoints(currentCellId, currIdList);

				// Find the point on the other end of the cell
				vtkIdType pA = currIdList->GetId(0);
				vtkIdType pB = currIdList->GetId(1);
				vtkIdType nextPoint = (pA == fromPoint) ? pB : pA;

				// If nextPoint is a junction, stop the branch here
				if (junctionPoints.count(nextPoint))
					continue;

				// Find unvisited neighboring cells connected to nextPoint
				const auto& neighborCells = pointToCells[nextPoint];
				for (vtkIdType neighborCellId : neighborCells)
				{
					if (!visited[neighborCellId])
					{
						q.push({ neighborCellId, nextPoint });
					}
				}
			}

			// After one traversal from an endpoint, increment BranchId
			currentBranchId++;
		}
	}

	input->GetCellData()->AddArray(branchIds);
}

float getAvgClosestDistance(vtkPolyData* triSkeleton, vtkPolyData* vortexLines) {

	// Build a locator for skeleton
	vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
	pointLocator->SetDataSet(triSkeleton);
	pointLocator->Update();

	float avgDist = 0.0;
	double pt[3];
	double closestPt[3];
	for (int i = 0; i < vortexLines->GetNumberOfPoints(); i++)
	{
		vortexLines->GetPoint(i, pt);
		vtkIdType ptId = pointLocator->FindClosestPoint(pt);
		triSkeleton->GetPoint(ptId, closestPt);
		avgDist += vtkMath::Distance2BetweenPoints(pt, closestPt);
	}

	avgDist = avgDist / vortexLines->GetNumberOfPoints();
	return avgDist;
}

void getCandidateSegments(vtkDataSet* dataset, vtkPolyData* triSkeleton, vtkPolyData* vortexLines,
	std::vector<vtkSmartPointer<vtkPolyData>>& candidateSegments)
{
	std::unordered_map<vtkIdType, std::vector<vtkIdType>> graph;
	buildAdjacencyList(triSkeleton, graph);
	cout << "a";
	vtkIdType minPtId = -1;
	vtkSmartPointer<vtkIdList> endPointIds = vtkSmartPointer<vtkIdList>::New();
	findEndPointsTriangle(triSkeleton, minPtId, endPointIds);
	cout << "b";
	AssignBranchIdsWithJunctionSplitting(triSkeleton);
	cout << "c";
	vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
	pointLocator->SetDataSet(dataset);
	pointLocator->Update();

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// Interpolate point data
	vtkSmartPointer<vtkGaussianKernel> kernel = vtkSmartPointer<vtkGaussianKernel>::New();
	kernel->SetKernelFootprintToNClosest();
	kernel->SetNumberOfPoints(32);

	vtkSmartPointer<vtkPointInterpolator> interpolator = vtkSmartPointer<vtkPointInterpolator>::New();
	interpolator->SetInputData(triSkeleton);
	interpolator->SetSourceData(dataset);
	interpolator->SetLocator(pointLocator);
	interpolator->SetKernel(kernel);
	interpolator->SetNullPointsStrategyToClosestPoint();
	interpolator->Update();
	triSkeleton->GetPointData()->DeepCopy(interpolator->GetPolyDataOutput()->GetPointData());
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	interpolatorTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	float minAvgDist = VTK_FLOAT_MAX;
	vtkIdType startId, endId;
	for (int i = 0; i < endPointIds->GetNumberOfIds(); i++) {
		startId = endPointIds->GetId(i);
		for (int j = i + 1; j < endPointIds->GetNumberOfIds(); j++)
		{
			endId = endPointIds->GetId(j);

			/*
			vtkSmartPointer<vtkDijkstraGraphGeodesicPath> dijkstra = vtkSmartPointer<vtkDijkstraGraphGeodesicPath>::New();
			dijkstra->SetInputData(triSkeleton);
			dijkstra->SetStartVertex(startId);
			dijkstra->SetEndVertex(endId);
			dijkstra->Update();
			*/

			// std::vector<vtkIdType> longestPath;
			std::vector<vtkIdType> currentPath;
			std::unordered_set<vtkIdType> visited;
			std::vector<std::vector<vtkIdType>> allPaths;
			// dfsLongest(startId, endId, graph, visited, currentPath, longestPath);
			// longestPath = bfsShortestPath(startId, endId, graph);
			dfsAllPaths(startId, endId, graph, visited, currentPath, allPaths);

			for (std::vector<vtkIdType> longestPath : allPaths) {
				vtkSmartPointer<vtkPolyData> longestSkel = vtkSmartPointer<vtkPolyData>::New();
				buildLinesFromPath(longestPath, triSkeleton, longestSkel);

				// We separate the spanwise (head) and streamwise (legs) sections of the skeleton
				vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
				triangleFilter->SetInputData(longestSkel);
				triangleFilter->Update();

				if (triangleFilter->GetOutput()->GetNumberOfPoints() > 0 &&
					triangleFilter->GetOutput()->GetNumberOfCells() > 0) {

					longestSkel->DeepCopy(triangleFilter->GetOutput());

					if (!checkSegment(longestSkel)) {
						continue;
					}

					vtkSmartPointer<ttkGeometrySmoother> smooth = vtkSmartPointer<ttkGeometrySmoother>::New();
					smooth->SetInputData(longestSkel);
					smooth->SetNumberOfIterations(5);
					smooth->Update();

					longestSkel->DeepCopy(smooth->GetOutput());

					candidateSegments.push_back(longestSkel);
					/*
					float avgDist = getAvgClosestDistance(longestSkel, vortexLines);
					if (avgDist < minAvgDist) {
						minAvgDist = avgDist;
						if (candidateSegments.size() > 0) {
							candidateSegments.pop_back();
							candidateSegments.push_back(longestSkel);
						}
						else
						{
							candidateSegments.push_back(longestSkel);
						}
					}
					*/
				}
			}
		}
	}
}

void visualizeSkel(vtkPolyData* skel, vtkPolyData* surface, vtkPolyData* isosurface, vtkRenderer* ren1, vtkRenderWindow* ren_win)
{
	// vtkSmartPointer<vtkPolyData> projSkeleton = vtkSmartPointer<vtkPolyData>::New();
	// getProjectedSkeleton(skel, projSkeleton);
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// Create the color transfer function
	vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
	colorTransferFunction->AddRGBPoint(0.0, 0.0, 0.0, 1.0); // Blue for minValue
	colorTransferFunction->AddRGBPoint(1.0, 1.0, 0.0, 0.0); // Red for maxValue
	colorTransferFunction->AddRGBPoint(2.0, 0.0, 1.0, 0.0); // Green for maxValue
	colorTransferFunction->AddRGBPoint(3.0, 1.0, 1.0, 1.0); // Green for maxValue
	colorTransferFunction->Build();

	vtkNew<vtkDataSetMapper> skelMapper1;
	skelMapper1->SetInputData(skel);
	skelMapper1->ScalarVisibilityOn();
	skelMapper1->SetScalarModeToUseCellData();
	skelMapper1->SetLookupTable(colorTransferFunction);
	skelMapper1->SetColorModeToMapScalars();
	skelMapper1->Update();

	vtkSmartPointer<vtkActor> skelActor1 = vtkSmartPointer<vtkActor>::New();
	skelActor1->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
	skelActor1->GetProperty()->SetLineWidth(10.0);
	skelActor1->GetProperty()->SetOpacity(1);
	skelActor1->SetMapper(skelMapper1);
	ren1->AddActor(skelActor1);

	surface->GetCellData()->SetActiveScalars("RegionIds");
	surface->GetCellData()->GetScalars()->Modified();
	// Create a lookup table to map cell data to colors.
	vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
	int tableSize = surface->GetCellData()->GetScalars()->GetRange()[1];
	lut->SetNumberOfTableValues(tableSize);
	lut->Build();
	RandomColors(lut, tableSize);

	vtkSmartPointer<vtkDataSetMapper> surfaceMapper1 = vtkSmartPointer<vtkDataSetMapper>::New();
	surfaceMapper1->SetInputData(surface);
	surfaceMapper1->ScalarVisibilityOff();
	// surfaceMapper1->SetScalarModeToUseCellData();
	// surfaceMapper1->SetColorModeToMapScalars();
	// surfaceMapper1->SetScalarRange(surface->GetCellData()->GetScalars()->GetRange());
	// surfaceMapper1->SetLookupTable(lut);
	surfaceMapper1->Update();

	vtkSmartPointer<vtkActor> surfaceActor1 = vtkSmartPointer<vtkActor>::New();
	surfaceActor1->SetMapper(surfaceMapper1);
	surfaceActor1->GetProperty()->SetColor(colors->GetColor3d("SkyBlue").GetData());
	surfaceActor1->GetProperty()->SetOpacity(0.45);
	ren1->AddActor(surfaceActor1);

	vtkSmartPointer<vtkDataSetMapper> isosurfaceMapper = vtkSmartPointer<vtkDataSetMapper>::New();
	isosurfaceMapper->SetInputData(isosurface);
	isosurfaceMapper->ScalarVisibilityOff();
	isosurfaceMapper->Update();

	vtkSmartPointer<vtkActor> isosurfaceActor = vtkSmartPointer<vtkActor>::New();
	isosurfaceActor->SetMapper(isosurfaceMapper);
	isosurfaceActor->GetProperty()->SetOpacity(0.25);
	isosurfaceActor->GetProperty()->SetColor(colors->GetColor3d("White").GetData());
	ren1->AddActor(isosurfaceActor);

	ren1->ResetCamera();

	vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> irenStyle =
		vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	iren->SetInteractorStyle(irenStyle);
	iren->SetRenderWindow(ren_win);

	ren_win->Render();
	iren->Start();

	ren1->RemoveActor(skelActor1);
	ren1->RemoveActor(surfaceActor1);
	ren1->RemoveActor(isosurfaceActor);
}

std::string vtkIdType_to_string(std::set<vtkIdType>& uniqueRegionIds) {
	std::string returnstring = "";
	for (vtkIdType uniqueRegionId : uniqueRegionIds)
		for (vtkIdType uniqueRegionId : uniqueRegionIds)
			returnstring += std::to_string(uniqueRegionId);
	return returnstring;
}

bool getOverlappingRegions(vtkPolyData* streamlines,
	vtkSmartPointer<vtkUnstructuredGrid> jointRegions)
{
	// We count how many streamlines are passing through each segment.
	vtkDataArray* regionIdsArray = DATASET->GetCellData()->GetArray("RegionIds");
	std::set<vtkIdType> uniqueRegionIds;

	/*
	for (int j = 0; j < streamlines->GetNumberOfPoints(); j++) {
		double point[3];
		streamlines->GetPoint(j, point);
		vtkIdType cellId = CellLocator->FindCell(point);
		vtkIdType regId = regionIdsArray->GetTuple1(cellId);
		uniqueRegionIds.insert(regId);
	}
	*/

	// Thread-local storage
	vtkSMPThreadLocal<std::unordered_set<vtkIdType>> threadLocalSets;
	auto regionIdRange = vtk::DataArrayTupleRange<1>(regionIdsArray);
	double test_pt[3];
	streamlines->GetPoint(0, test_pt);

	vtkSMPTools::For(0, streamlines->GetNumberOfPoints(), [&](vtkIdType start, vtkIdType end) {

		auto& localSet = threadLocalSets.Local();  // Each thread gets its own map
		double point[3];
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New(); // thread-local cell
		double pcoords[3];
		double weights[8];

		for (vtkIdType j = start; j < end; ++j) {
			streamlines->GetPoint(j, point);

			vtkIdType cellId = CellLocator->FindCell(point, CellLocator->GetTolerance(),
				localCell, pcoords, weights);

			if (cellId >= 0) {
				vtkIdType regId = static_cast<vtkIdType>(regionIdRange[cellId][0]);
				localSet.insert(regId);
			}
		}
		});

	// Final merge of all thread-local sets into the output
	for (auto it = threadLocalSets.begin(); it != threadLocalSets.end(); ++it) {
		for (const vtkIdType& pair : *it) {
			uniqueRegionIds.insert(pair);
		}
	}

	std::string segIdsString = vtkIdType_to_string(uniqueRegionIds);
	if (regionStrings.count(segIdsString)) {
		return false;
	}
	regionStrings.insert(segIdsString);

	// Count total number of unique ids that have already been visited.
	int countVisited = 0;
	int countTotal = uniqueRegionIds.size();
	for (vtkIdType regionId : uniqueRegionIds) {
		if (procesdRegions.count(regionId) > 0) {
			countVisited++;
		}
	}
	float percentageVisited = (float)countVisited / (float)countTotal;
	// Return false if more than 50% regions are already visisted.
	if (percentageVisited > 0.5) {
		return false;
	}

	// If the number of streanlines passing through each segment are less than 'cellTh', then we don't consider that segment.
	vtkSmartPointer<vtkAppendFilter> append = vtkSmartPointer<vtkAppendFilter>::New();
	append->MergePointsOn();
	int countRegs = 0;
	for (vtkIdType regionId : uniqueRegionIds)
	{
		countRegs++;
		// cout << regionId << " ";
		append->AddInputData(regionsMap[regionId]);
	}
	if (countRegs == 0)
	{
		return false;
	}
	append->Update();
	jointRegions->DeepCopy(append->GetOutput());

	// cout << endl;
	if (jointRegions->GetNumberOfCells() == 0)
	{
		return false;
	}

	// shredRegion(jointRegions);
	return true;
}

bool checkRegionIdSkel(vtkPolyData* skel, std::unordered_map<int, std::unordered_set<int>>& segToSkelMap, vtkIdType& regionId)
{
	vtkDataArray* branchIds = skel->GetCellData()->GetArray("BranchIds");
	for (vtkIdType i = 0; i < skel->GetNumberOfCells(); ++i)
	{
		int branchId = branchIds->GetTuple1(i);
		if (segToSkelMap.count(branchId) == 0) {
			continue;
		}
		for (int regId : segToSkelMap[branchId]) {
			if (regId == regionId) {
				return true;
			}
			else {
				continue;
			}
		}
	}
	return false;
}

void cleanSkelRegions(vtkPolyData* streamlines, vtkUnstructuredGrid* jointRegions) {

	vtkSmartPointer<vtkPolyData> surface = vtkSmartPointer<vtkPolyData>::New();
	extractSurface(jointRegions, surface);

	vtkSmartPointer<vtkCellLocator> locator = vtkSmartPointer<vtkCellLocator>::New();
	locator->SetDataSet(streamlines);
	locator->BuildLocator();

	vtkSmartPointer<vtkCellCenters> cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(surface);
	cellCenters->Update();
	vtkPolyData* input = cellCenters->GetOutput();

	vtkIdList* pointIds = vtkIdList::New();
	std::map<vtkIdType, vtkIdType> globalToLocalPointMap;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> cellsArray = vtkSmartPointer<vtkCellArray>::New();

	double avgDist = 0;
	for (vtkIdType cellId = 0; cellId < surface->GetNumberOfCells(); ++cellId) {

		double queryPt[3];
		double closestPt[3];
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New();
		vtkIdType id;
		int subId;
		double dist2;

		input->GetPoint(cellId, queryPt);

		locator->FindClosestPoint(queryPt, closestPt, localCell, id, subId, dist2);
		avgDist += dist2;
	}
	avgDist /= input->GetNumberOfPoints();


	cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(jointRegions);
	cellCenters->Update();
	input = cellCenters->GetOutput();

	for (vtkIdType cellId = 0; cellId < jointRegions->GetNumberOfCells(); ++cellId) {

		double queryPt[3];
		double closestPt[3];
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New();
		vtkIdType id;
		int subId;
		double dist2;

		input->GetPoint(cellId, queryPt);

		locator->FindClosestPoint(queryPt, closestPt, localCell, id, subId, dist2);

		if (dist2 > avgDist) {
			continue;
		}

		jointRegions->GetCellPoints(cellId, pointIds);

		std::vector<vtkIdType> localPtIds(pointIds->GetNumberOfIds());

		for (vtkIdType j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			vtkIdType globalId = pointIds->GetId(j);

			if (globalToLocalPointMap.find(globalId) == globalToLocalPointMap.end()) {
				vtkIdType newId = points->InsertNextPoint(jointRegions->GetPoint(globalId));
				globalToLocalPointMap[globalId] = newId;
			}

			localPtIds[j] = globalToLocalPointMap[globalId];
		}

		cellsArray->InsertNextCell(static_cast<vtkIdType>(localPtIds.size()), localPtIds.data());
	}


	// Assemble final vtkUnstructuredGrid objects
	vtkSmartPointer<vtkUnstructuredGrid> newJointRegions = vtkSmartPointer<vtkUnstructuredGrid>::New();
	newJointRegions->SetPoints(points);
	newJointRegions->SetCells(jointRegions->GetCellType(0), cellsArray);

	vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
	pointLocator->SetDataSet(jointRegions);
	pointLocator->BuildLocator();

	vtkSmartPointer<vtkPointInterpolator> interpolator = vtkSmartPointer<vtkPointInterpolator>::New();
	interpolator->SetInputData(newJointRegions);     // Region grid with only geometry
	interpolator->SetSourceData(jointRegions);
	interpolator->SetLocator(pointLocator);
	// Use nearest-neighbor interpolation
	vtkSmartPointer<vtkVoronoiKernel> voronoiKernel = vtkSmartPointer<vtkVoronoiKernel>::New();
	interpolator->SetKernel(voronoiKernel);
	interpolator->SetNullPointsStrategyToClosestPoint(); // Use closest point if no exact match
	interpolator->Update();
	newJointRegions->DeepCopy(interpolator->GetOutput());
	newJointRegions->Squeeze();

	pointIds->Delete();

	vtkSmartPointer<vtkGenericCell> tempCell = vtkSmartPointer<vtkGenericCell>::New();
	vtkCellData* inputCellData = jointRegions->GetCellData();
	int numCellArrays = inputCellData->GetNumberOfArrays();

	vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();
	cellLocator->SetDataSet(jointRegions);
	cellLocator->BuildLocator();

	// Prepare new cell data arrays for the region
	std::vector<vtkSmartPointer<vtkDataArray>> regionCellArrays(numCellArrays);
	for (int i = 0; i < numCellArrays; ++i) {
		vtkDataArray* sourceArray = inputCellData->GetArray(i);
		auto newArray = vtkSmartPointer<vtkDataArray>::Take(sourceArray->NewInstance());
		newArray->SetName(sourceArray->GetName());
		newArray->SetNumberOfComponents(sourceArray->GetNumberOfComponents());
		newArray->SetNumberOfTuples(newJointRegions->GetNumberOfCells());
		regionCellArrays[i] = newArray;
	}

	// For each cell in the region, find closest in original and copy values manually
	for (vtkIdType cid = 0; cid < newJointRegions->GetNumberOfCells(); ++cid) {
		double center[3] = { 0, 0, 0 };
		vtkCell* cell = newJointRegions->GetCell(cid);
		double pt[3];
		for (vtkIdType i = 0; i < cell->GetNumberOfPoints(); ++i) {
			newJointRegions->GetPoint(cell->GetPointId(i), pt);
			center[0] += pt[0];
			center[1] += pt[1];
			center[2] += pt[2];
		}
		center[0] /= cell->GetNumberOfPoints();
		center[1] /= cell->GetNumberOfPoints();
		center[2] /= cell->GetNumberOfPoints();

		vtkIdType closestCellId = cellLocator->FindCell(center);
		if (closestCellId < 0) continue;

		// Copy each array manually
		for (int i = 0; i < numCellArrays; ++i) {
			vtkDataArray* sourceArray = inputCellData->GetArray(i);
			vtkDataArray* destArray = regionCellArrays[i];

			int numComps = sourceArray->GetNumberOfComponents();
			std::vector<double> tuple(numComps);
			sourceArray->GetTuple(closestCellId, tuple.data());
			destArray->SetTuple(cid, tuple.data());
		}
	}

	// Attach all cell arrays to regionGrid
	for (auto& array : regionCellArrays) {
		array->Squeeze();
		newJointRegions->GetCellData()->AddArray(array);
	}

	jointRegions->DeepCopy(newJointRegions);
}

void getOverlappingRegionsSkel(vtkPolyData* skel, vtkSmartPointer<vtkUnstructuredGrid> skelRegions,
	vtkSmartPointer<vtkUnstructuredGrid> jointRegions, std::set<vtkIdType>& uniqueRegionIds,
	std::unordered_map<int, std::unordered_set<int>>& segToSkelMap)
{
	vtkDataArray* branchIds = skel->GetCellData()->GetArray("BranchIds");
	for (int i = 0; i < skel->GetNumberOfCells(); ++i)
	{
		int branchId = branchIds->GetTuple1(i);
		if (segToSkelMap.count(branchId) == 0) {
			continue;
		}
		for (int j : segToSkelMap[branchId]) {
			uniqueRegionIds.insert(j);
		}
	}

	vtkSmartPointer<vtkAppendFilter> append = vtkSmartPointer<vtkAppendFilter>::New();
	append->MergePointsOn();
	for (int regionId : uniqueRegionIds)
	{
		vtkSmartPointer<vtkThreshold> regThreshold = vtkSmartPointer<vtkThreshold>::New();
		regThreshold->SetInputData(jointRegions);
		regThreshold->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "SegmentIds");
		regThreshold->SetLowerThreshold(regionId);
		regThreshold->SetUpperThreshold(regionId);
		regThreshold->Update();
		append->AddInputData(regThreshold->GetOutput());
	}
	append->Update();

	vtkSmartPointer<vtkConnectivityFilter> isoConnFil = vtkSmartPointer<vtkConnectivityFilter>::New();
	isoConnFil->SetInputData(append->GetOutput());
	isoConnFil->SetExtractionModeToLargestRegion();
	isoConnFil->ColorRegionsOff();
	isoConnFil->Update();

	skelRegions->DeepCopy(isoConnFil->GetOutput());

	uniqueRegionIds.clear();
	vtkDataArray* regionIds = skelRegions->GetCellData()->GetArray("RegionIds");
	for (int i = 0; i < regionIds->GetNumberOfTuples(); ++i) {
		uniqueRegionIds.insert(regionIds->GetTuple1(i));
	}
}

void getImmediateNeighborsFace(vtkIdType id, vtkIdList* neighborCells)
{
	neighborCells->Reset();
	// vtkNew<vtkGenericCell> cell;
	// dataset->GetCell(id, cell);
	vtkCell* cell = DATASET->GetCell(id);
	vtkNew<vtkIdList> neighborCellIds, pointCells;
	for (int faceId = 0; faceId < cell->GetNumberOfFaces(); faceId++)
	{
		pointCells->Reset();
		neighborCellIds->Reset();
		vtkCell* face = cell->GetFace(faceId);
		vtkIdList* facePoints = face->GetPointIds();
		for (int ptId = 0; ptId < facePoints->GetNumberOfIds(); ptId++)
		{
			pointCells->InsertNextId(facePoints->GetId(ptId));
		}
		DATASET->GetCellNeighbors(id, pointCells, neighborCellIds);
		for (vtkIdType j = 0; j < neighborCellIds->GetNumberOfIds(); j++)
		{
			neighborCells->InsertNextId(neighborCellIds->GetId(j));
		}
	}
	// neighborCells->Sort();
}

void labelSurfaceParts(vtkPolyData* skeleton, vtkPolyData* surface)
{
	vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();
	cellLocator->SetDataSet(skeleton);
	cellLocator->BuildLocator();

	vtkSmartPointer<vtkCellCenters> cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(surface);
	cellCenters->Update();

	vtkSmartPointer<vtkPolyData> centers = cellCenters->GetOutput();

	// Add a cell array to keep track of spanwise(1) and streamwise(0) segments.
	vtkSmartPointer<vtkUnsignedCharArray> surfaceCellDirArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
	surfaceCellDirArray->SetNumberOfComponents(1);
	surfaceCellDirArray->SetNumberOfTuples(surface->GetNumberOfCells());
	surfaceCellDirArray->SetName("VortexParts");

	vtkDataArray* skelCellDirArr = skeleton->GetCellData()->GetArray("cellDir");
	for (int i = 0; i < surface->GetNumberOfCells(); i++)
	{
		double pt[3];
		centers->GetPoint(i, pt);
		double closestPoint[3];
		vtkIdType cellId;
		int subId;
		double dist2;
		cellLocator->FindClosestPoint(pt, closestPoint, cellId, subId, dist2);
		surfaceCellDirArray->SetTuple1(i, skelCellDirArr->GetTuple1(cellId));
	}
	surface->GetCellData()->AddArray(surfaceCellDirArray);
}

void labelRegionParts(vtkPolyData* skeleton, vtkUnstructuredGrid* region)
{
	vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();
	cellLocator->SetDataSet(skeleton);
	cellLocator->BuildLocator();

	// Add a point array to keep track of spanwise(1) and streamwise(0) segments.
	vtkSmartPointer<vtkUnsignedCharArray> regionPointDirArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
	regionPointDirArray->SetNumberOfComponents(1);
	regionPointDirArray->SetNumberOfTuples(region->GetNumberOfPoints());
	regionPointDirArray->SetName("VortexParts");

	vtkDataArray* skelCellDirArr = skeleton->GetCellData()->GetArray("cellDir");
	for (int i = 0; i < region->GetNumberOfPoints(); i++)
	{
		double pt[3];
		region->GetPoint(i, pt);
		double closestPoint[3];
		vtkIdType cellId;
		int subId;
		double dist2;
		cellLocator->FindClosestPoint(pt, closestPoint, cellId, subId, dist2);
		regionPointDirArray->SetTuple1(i, skelCellDirArr->GetTuple1(cellId));
	}
	region->GetPointData()->AddArray(regionPointDirArray);
}

double ComputeBoundingBoxOverlap(const double boundsA[6], const double boundsB[6])
{
	// Compute overlap along each axis
	double xOverlap = std::max(0.0, std::min(boundsA[1], boundsB[1]) - std::max(boundsA[0], boundsB[0]));
	double yOverlap = std::max(0.0, std::min(boundsA[3], boundsB[3]) - std::max(boundsA[2], boundsB[2]));
	double zOverlap = std::max(0.0, std::min(boundsA[5], boundsB[5]) - std::max(boundsA[4], boundsB[4]));

	// If there's no overlap in any dimension, the overlap volume is zero
	double overlapVolume = xOverlap * yOverlap * zOverlap;
	if (overlapVolume <= 0.0)
		return 0.0;

	// Compute individual volumes
	double volA = (boundsA[1] - boundsA[0]) * (boundsA[3] - boundsA[2]) * (boundsA[5] - boundsA[4]);
	double volB = (boundsB[1] - boundsB[0]) * (boundsB[3] - boundsB[2]) * (boundsB[5] - boundsB[4]);

	if (volA <= 0.0 || volB <= 0.0)
		return 0.0;

	// Compute overlap ratio (e.g., Intersection over Union)
	double unionVolume = volA + volB - overlapVolume;
	return overlapVolume / unionVolume; // value in [0, 1]
}

float getNearestBoundingBox(const double boundsA[6], const double boundsB[6]) {
	float boundsACenter[3] = { (boundsA[0] + boundsA[1]) / 2, (boundsA[2] + boundsA[3]) / 2, (boundsA[4] + boundsA[5]) / 2 };
	float boundsBCenter[3] = { (boundsB[0] + boundsB[1]) / 2, (boundsB[2] + boundsB[3]) / 2, (boundsB[4] + boundsB[5]) / 2 };
	return vtkMath::Distance2BetweenPoints(boundsACenter, boundsBCenter);
}

void findClosestBranches(vtkUnstructuredGrid* jointRegions, vtkPolyData* skeleton,
	std::unordered_map<int, std::unordered_set<int>>& segToSkelMap, std::unordered_map<int, std::unordered_set<int>>& regToSkelMap)
{
	// Build the locator
	vtkSmartPointer<vtkCellLocator> locator = vtkSmartPointer<vtkCellLocator>::New();
	locator->SetDataSet(skeleton);
	locator->BuildLocator();

	vtkDataArray* branchIdsAr = skeleton->GetCellData()->GetArray("BranchIds");
	vtkDataArray* segmentIdsAr = jointRegions->GetCellData()->GetArray("SegmentIds");

	auto branchIdsRange = vtk::DataArrayTupleRange<1>(branchIdsAr);
	auto segmentIdsRange = vtk::DataArrayTupleRange<1>(segmentIdsAr);

	vtkSmartPointer<vtkCellCenters> cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(jointRegions);
	cellCenters->Update();
	vtkPolyData* input = cellCenters->GetOutput();

	vtkIdType numPoints = input->GetNumberOfPoints();

	for (vtkIdType i = 0; i < numPoints; ++i) {
		double queryPt[3];
		double closestPt[3];
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New();
		vtkIdType cellId;
		int subId;
		double dist2;

		input->GetPoint(i, queryPt);

		locator->FindClosestPoint(queryPt, closestPt, localCell, cellId, subId, dist2);
		int branchId = static_cast<int>(branchIdsAr->GetTuple1(cellId));
		int segmentId = static_cast<int>(segmentIdsAr->GetTuple1(i));

		segToSkelMap[branchId].insert(segmentId);
	}

	vtkDataArray* regionIdsAr = jointRegions->GetCellData()->GetArray("RegionIds");
	for (vtkIdType i = 0; i < numPoints; ++i) {
		double queryPt[3];
		double closestPt[3];
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New();
		vtkIdType cellId;
		int subId;
		double dist2;

		input->GetPoint(i, queryPt);

		locator->FindClosestPoint(queryPt, closestPt, localCell, cellId, subId, dist2);
		int branchId = static_cast<int>(branchIdsAr->GetTuple1(cellId));
		int regionId = static_cast<int>(regionIdsAr->GetTuple1(i));

		regToSkelMap[branchId].insert(regionId);
	}
}

void fixClosestBranches(vtkUnstructuredGrid* jointRegions, vtkPolyData* skeleton,
	std::unordered_map<int, std::unordered_set<int>>& segToSkelMap, double& avgDist) {
	// Build the locator
	vtkSmartPointer<vtkCellLocator> locator = vtkSmartPointer<vtkCellLocator>::New();
	locator->SetDataSet(skeleton);
	locator->BuildLocator();

	vtkDataArray* branchIdsAr = skeleton->GetCellData()->GetArray("BranchIds");
	vtkDataArray* segmentIdsAr = jointRegions->GetCellData()->GetArray("RegionIds");

	vtkSmartPointer<vtkCellCenters> cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(jointRegions);
	cellCenters->Update();
	vtkPolyData* input = cellCenters->GetOutput();

	double queryPt[3];
	for (int i = 0; i < input->GetNumberOfPoints(); ++i) {
		input->GetPoint(i, queryPt);

		// Prepare output variables
		double closestPt[3];
		vtkIdType cellId;
		int subId;
		double dist2;

		// Query
		locator->FindClosestPoint(queryPt, closestPt, cellId, subId, dist2);
		if (dist2 < avgDist) {
			int branchId = branchIdsAr->GetTuple1(cellId);
			int segmentId = segmentIdsAr->GetTuple1(i);
			segToSkelMap[branchId].insert(segmentId);
		}
	}
}

void extractContours(vtkDataSet* jointRegions, vtkPolyData* isosurface) {
	vtkSmartPointer<vtkContourFilter> contourFilter = vtkSmartPointer<vtkContourFilter>::New();
	contourFilter->SetInputData(jointRegions);
	contourFilter->SetValue(0, -13.3954);
	contourFilter->Update();

	isosurface->DeepCopy(contourFilter->GetOutput());
}

void getManifoldRegion(vtkDataSet* dataset)
{
	vtkSmartPointer<vtkDataSetTriangleFilter> tetrahedralize =
		vtkSmartPointer<vtkDataSetTriangleFilter>::New();
	tetrahedralize->SetInputData(dataset);
	tetrahedralize->Update();

	vtkSmartPointer<ttkManifoldCheck> manCheck = vtkSmartPointer<ttkManifoldCheck>::New();
	manCheck->SetInputData(tetrahedralize->GetOutput());
	manCheck->Update();

	vtkSmartPointer<vtkThreshold> manThreshold = vtkSmartPointer<vtkThreshold>::New();
	manThreshold->SetInputData(manCheck->GetOutput());
	manThreshold->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "VertexLinkComponentNumber");
	manThreshold->SetLowerThreshold(1);
	manThreshold->SetUpperThreshold(1);
	manThreshold->Update();

	vtkSmartPointer<vtkConnectivityFilter> connFilter = vtkSmartPointer<vtkConnectivityFilter>::New();
	connFilter->SetInputData(manThreshold->GetOutput());
	connFilter->SetExtractionModeToLargestRegion();
	connFilter->ColorRegionsOff();
	connFilter->Update();

	dataset->DeepCopy(connFilter->GetOutput());

	//Remove unnecessary arrays
	dataset->GetCellData()->RemoveArray("VertexLinkComponentNumber");
	dataset->GetCellData()->RemoveArray("EdgeLinkComponentNumber");
	dataset->GetCellData()->RemoveArray("TriangleLinkComponentNumber");
	dataset->GetPointData()->RemoveArray("VertexLinkComponentNumber");
	dataset->GetPointData()->RemoveArray("EdgeLinkComponentNumber");
	dataset->GetPointData()->RemoveArray("TriangleLinkComponentNumber");
	dataset->Modified();
}

bool findAndJoinRegions(vtkPolyData* newSkeleton, vtkPolyData* newSurface, vtkPolyData* complete_surface_data,
	vtkPolyData* vortex_lines_data, vtkUnstructuredGrid* regions_data, vtkUnstructuredGrid* complete_regions_data,
	vtkPolyData* skeleton, vtkPolyData* isosurface_data, vtkIdType& regionId, int expectedType, int& counter)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vtkUnstructuredGrid* source = regionsMap[regionId];

	cout << "A";
	vtkSMPTools::SetBackend("STDThread");
	vtkSmartPointer<vtkStreamTracer> Streamers = vtkSmartPointer<vtkStreamTracer>::New();
	Streamers->SetInputData(DATASET);
	Streamers->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "vorticity");
	Streamers->SetInterpolatorTypeToCellLocator();
	Streamers->SetIntegratorTypeToRungeKutta45();
	Streamers->SetIntegrationDirectionToBoth();
	Streamers->SetMaximumPropagation(DATASET->GetLength());
	Streamers->SetComputeVorticity(0);
	Streamers->SetRotationScale(0);
	Streamers->SetSourceData(source);
	Streamers->Update();
	vtkPolyData* streamlines1 = Streamers->GetOutput();
	vtkSMPTools::SetBackend("Sequential");

	cout << "B";
	bool foundRegions;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	streamTracerTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	begin = std::chrono::steady_clock::now();
	vtkSmartPointer<vtkUnstructuredGrid> jointRegions = vtkSmartPointer<vtkUnstructuredGrid>::New();
	foundRegions = getOverlappingRegions(streamlines1, jointRegions);
	end = std::chrono::steady_clock::now();
	findCandidatesTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	cout << "C";
	if (foundRegions == false)
	{
		visitedRegions.insert(regionId);
		return false;
	}

	// getManifoldRegion(jointRegions);
	begin = std::chrono::steady_clock::now();
	vtkSmartPointer<vtkPolyData> surface = vtkSmartPointer<vtkPolyData>::New();
	extractSurface(jointRegions, surface);
	vtkSmartPointer<vtkPolyData> isosurface = vtkSmartPointer<vtkPolyData>::New();
	extractIsosurface(surface, isosurface);
	// extractContours(jointRegions, isosurface);
	constructSurface(jointRegions, isosurface, surface);

	end = std::chrono::steady_clock::now();
	surfaceExtractionTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	cout << "D";
	begin = std::chrono::steady_clock::now();

	extractSkeleton(surface, skeleton);

	cout << "E";
	// extractSkeleton2(jointRegions, skeleton);

	end = std::chrono::steady_clock::now();
	skelExtractionTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	if (skeleton->GetNumberOfCells() == 0 ||
		skeleton->GetNumberOfPoints() == 0)
	{
		visitedRegions.insert(regionId);
		return false;
	}
	begin = std::chrono::steady_clock::now();
	std::vector<vtkSmartPointer<vtkPolyData>> candidateSegments;
	getCandidateSegments(jointRegions, skeleton, streamlines1, candidateSegments);
	cout << "F";
	if (candidateSegments.size() == 0) {
		visitedRegions.insert(regionId);
		return false;
	}
	std::unordered_map<int, std::unordered_set<int>> segToSkelMap;
	std::unordered_map<int, std::unordered_set<int>> regToSkelMap;
	double avgDist = 0.0;
	findClosestBranches(jointRegions, skeleton, segToSkelMap, regToSkelMap);
	end = std::chrono::steady_clock::now();
	skelFindTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	cout << "G";

	/*
	vtkNew<vtkNamedColors> colors;
	vtkNew<vtkRenderer> ren1;
	ren1->SetBackground(colors->GetColor3d("White").GetData());
	vtkNew<vtkRenderWindow> ren_win;
	ren_win->SetSize(1200, 1000);
	ren_win->AddRenderer(ren1);

	vtkNew<vtkOutlineFilter> outline;
	outline->SetInputData(surface);
	outline->Update();
	vtkNew<vtkDataSetMapper> outlineMapper;
	outlineMapper->SetInputConnection(outline->GetOutputPort());
	vtkNew<vtkActor> outlineActor;
	outlineActor->SetMapper(outlineMapper);
	outlineActor->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
	outlineActor->GetProperty()->SetLineWidth(1.0);
	ren1->AddActor(outlineActor);

	vtkColor3d axesActorColor = colors->GetColor3d("Black");
	vtkNew<vtkCubeAxesActor> cubeAxesActor;
	cubeAxesActor->SetUseTextActor3D(0);
	cubeAxesActor->SetBounds(outline->GetOutput()->GetBounds());
	cubeAxesActor->SetLabelOffset(10);
	cubeAxesActor->SetTitleOffset(10);
	cubeAxesActor->SetLabelScaling(false, 0, 0, 0);
	cubeAxesActor->GetTitleTextProperty(0)->SetColor(axesActorColor.GetData());
	cubeAxesActor->GetLabelTextProperty(0)->SetColor(axesActorColor.GetData());
	cubeAxesActor->GetTitleTextProperty(1)->SetColor(axesActorColor.GetData());
	cubeAxesActor->GetLabelTextProperty(1)->SetColor(axesActorColor.GetData());
	cubeAxesActor->GetTitleTextProperty(2)->SetColor(axesActorColor.GetData());
	cubeAxesActor->GetLabelTextProperty(2)->SetColor(axesActorColor.GetData());
	cubeAxesActor->XAxisMinorTickVisibilityOff();
	cubeAxesActor->YAxisMinorTickVisibilityOff();
	cubeAxesActor->ZAxisMinorTickVisibilityOff();
	// cubeAxesActor->SetScreenSize(10.0);
	cubeAxesActor->SetFlyMode(vtkCubeAxesActor::VTK_FLY_OUTER_EDGES);
	cubeAxesActor->SetCamera(ren1->GetActiveCamera());
	ren1->AddActor(cubeAxesActor);

	vtkNew<vtkDataSetMapper> streamMapper1;
	streamMapper1->SetInputData(streamlines1);
	streamMapper1->ScalarVisibilityOff();
	streamMapper1->Update();
	vtkNew<vtkActor> streamActor1;
	streamActor1->SetMapper(streamMapper1);
	streamActor1->GetProperty()->SetLineWidth(1.0);
	streamActor1->GetProperty()->SetOpacity(0.25);
	streamActor1->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
	ren1->AddActor(streamActor1);

	ren1->GetActiveCamera()->SetPosition(-0.1, 0, 0.1);
	ren1->GetActiveCamera()->SetViewUp(0, 0, 0.25);
	ren1->GetActiveCamera()->Zoom(2.0);

	vtkNew<vtkDataSetMapper> skelMapper1;
	skelMapper1->SetInputData(skeleton);
	skelMapper1->ScalarVisibilityOff();
	// skelMapper1->SetScalarModeToUseCellData();
	// skelMapper1->SetLookupTable(colorTransferFunction);
	// skelMapper1->SetColorModeToMapScalars();
	skelMapper1->Update();

	vtkSmartPointer<vtkActor> skelActor1 = vtkSmartPointer<vtkActor>::New();
	skelActor1->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
	skelActor1->GetProperty()->SetLineWidth(5.0);
	skelActor1->GetProperty()->SetOpacity(0.75);
	skelActor1->SetMapper(skelMapper1);
	ren1->AddActor(skelActor1);

	visualizeSkel(skeleton, surface, isosurface, ren1, ren_win);
	*/

	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	triangleFilter->SetInputData(streamlines1);
	triangleFilter->Update();
	vtkPolyData* triStreamlines = triangleFilter->GetOutput();

	bool foundHairpin = false;
	float lenA, maxLen;
	float consistA, maxConsist;
	float hausDist, minHausDist;
	bool foundHead = false;
	int finalType;
	if (candidateSegments.size() > 0)
	{
		// cout << "Checking for hairpin vortex type..." << regionId << endl;
		begin = std::chrono::steady_clock::now();
		for (vtkSmartPointer<vtkPolyData> candidateSkel : candidateSegments)
		{
			int type = -1;
			candidateSkel->Modified();
			consistA = checkVortexConsistency(candidateSkel);

			if (consistA < CONSISTENCY) {
				continue;
			}

			// cout << " " << consistA << " " << hausDist << endl;
			bool check = false;
			checkForHairpinType(candidateSkel, type);
			// cout << type << ", ";
			if (type == expectedType) {
				check = true;
			}
			else if (type >= 1) {
				foundHead = true;
			}

			if (check) {
				bool foundRegion = checkRegionIdSkel(candidateSkel, regToSkelMap, regionId);
				if (foundRegion) {
					foundHairpin = true;
					if (newSkeleton->GetNumberOfCells() == 0) {
						newSkeleton->DeepCopy(candidateSkel);
						newSkeleton->Modified();
						minHausDist = getHausDist(candidateSkel, triStreamlines);
						maxLen = getSkeletonLength(newSkeleton);
						maxConsist = consistA;
						finalType = type;
					}
					else {
						lenA = getSkeletonLength(candidateSkel);
						hausDist = getHausDist(candidateSkel, triStreamlines);
						if (lenA > maxLen) {
							newSkeleton->DeepCopy(candidateSkel);
							newSkeleton->Modified();
							maxConsist = consistA;
							minHausDist = hausDist;
							finalType = type;
							maxLen = lenA;
						}
					}
				}
			}
		}

		end = std::chrono::steady_clock::now();
		postProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

		if (foundHairpin == true) {
			begin = std::chrono::steady_clock::now();

			/*
			bool foundBetter = verifyHairpin(newSkeleton, newSurface, complete_surface_data,
				vortex_lines_data, regions_data, complete_regions_data, skeleton, isosurface_data,
				regionId, maxLen, maxConsist);
			if (foundBetter) {
				return foundBetter;
			}
			*/

			// Recheck for a better type
			for (vtkSmartPointer<vtkPolyData> candidateSkel : candidateSegments)
			{
				int type = -1;
				candidateSkel->Modified();
				consistA = checkVortexConsistency(candidateSkel);

				if (consistA < CONSISTENCY) {
					continue;
				}

				bool check = false;
				checkForHairpinType(candidateSkel, type);

				if (type > 0) {
					bool foundRegion = checkRegionIdSkel(candidateSkel, regToSkelMap, regionId);
					if (foundRegion) {
						hausDist = getHausDist(candidateSkel, triStreamlines);
						if (hausDist * 2 < minHausDist)
						{
							newSkeleton->DeepCopy(candidateSkel);
							minHausDist = hausDist;
							finalType = type;
						}
					}
				}
			}

			vtkSmartPointer<vtkUnstructuredGrid> skelRegions = vtkSmartPointer<vtkUnstructuredGrid>::New();
			std::set<vtkIdType> uniqueRegionIds;
			// fixClosestBranches(jointRegions, newSkeleton, segToSkelMap, avgDist);

			getOverlappingRegionsSkel(newSkeleton, skelRegions, jointRegions, uniqueRegionIds, segToSkelMap);
			// cleanSkelRegions(streamlines1, skelRegions);

			if (uniqueRegionIds.size() <= 1 || uniqueRegionIds.count(regionId) == 0) {
				visitedRegions.insert(regionId);
				return false;
			}

			getManifoldRegion(skelRegions);
			getNewSurface(skelRegions, newSurface);

			if (newSurface->GetNumberOfPoints() == 0)
			{
				return false;
			}
			extractIsosurface(newSurface, isosurface);
			// extractContours(skelRegions, isosurface);
			constructSurface(skelRegions, isosurface, newSurface);

			// addTypeArray(newSurface, expectedType);

			regions_data->DeepCopy(skelRegions);
			vortex_lines_data->DeepCopy(streamlines1);
			complete_regions_data->DeepCopy(jointRegions);
			complete_surface_data->DeepCopy(surface);
			isosurface_data->DeepCopy(isosurface);
			// skeleton->DeepCopy(skel);
			addTypeArray(newSkeleton, finalType);

			// Here we label the head, neck and legs part of the newly constructed surface
			// labelSurfaceParts(newSkeleton, newSurface);
			// labelRegionParts(newSkeleton, regions_data);

			cout << " Found hairpin vortex..." << finalType << endl;

			for (vtkIdType uniqueRegId : uniqueRegionIds) {
				procesdRegions.insert(uniqueRegId);
			}

			procesdHeads.insert(regionId);
			// visualizeSkel(newSkeleton, newSurface, surface, ren1, ren_win);
			end = std::chrono::steady_clock::now();
			postProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			return foundHairpin;
		}
	}

	if (!foundHead) {
		// This segment didn't result in any type so we don't need to check it the next time.
		visitedRegions.insert(regionId);
	}
	return foundHairpin;
}

void createCellData(std::unordered_map<vtkIdType, vtkSmartPointer<vtkUnstructuredGrid>>& map) {
	cout << "Computing cell data..." << endl;
	// Create a reusable cell object for getting data from DATASET
	vtkSmartPointer<vtkGenericCell> tempCell = vtkSmartPointer<vtkGenericCell>::New();
	vtkCellData* inputCellData = DATASET->GetCellData();
	int numCellArrays = inputCellData->GetNumberOfArrays();
	int i = 0;
	for (auto& pair : map) {
		int regionId = pair.first;
		if (i % 1000 == 0) {
			cout << i << " " << map.size() << endl;
		}
		i++;
		vtkUnstructuredGrid* grid = pair.second;
		// Prepare new cell data arrays for the region
		std::vector<vtkSmartPointer<vtkDataArray>> regionCellArrays(numCellArrays);
		for (int i = 0; i < numCellArrays; ++i) {
			vtkDataArray* sourceArray = inputCellData->GetArray(i);
			auto newArray = vtkSmartPointer<vtkDataArray>::Take(sourceArray->NewInstance());
			newArray->SetName(sourceArray->GetName());
			newArray->SetNumberOfComponents(sourceArray->GetNumberOfComponents());
			newArray->SetNumberOfTuples(grid->GetNumberOfCells());
			regionCellArrays[i] = newArray;
		}

		// For each cell in the region, find closest in original and copy values manually
		for (vtkIdType cid = 0; cid < grid->GetNumberOfCells(); ++cid) {
			double center[3] = { 0, 0, 0 };
			vtkCell* cell = grid->GetCell(cid);
			double pt[3];
			for (vtkIdType i = 0; i < cell->GetNumberOfPoints(); ++i) {
				grid->GetPoint(cell->GetPointId(i), pt);
				center[0] += pt[0];
				center[1] += pt[1];
				center[2] += pt[2];
			}
			center[0] /= cell->GetNumberOfPoints();
			center[1] /= cell->GetNumberOfPoints();
			center[2] /= cell->GetNumberOfPoints();

			vtkIdType closestCellId = CellLocator->FindCell(center);
			if (closestCellId < 0) continue;

			// Copy each array manually
			for (int i = 0; i < numCellArrays; ++i) {
				vtkDataArray* sourceArray = inputCellData->GetArray(i);
				vtkDataArray* destArray = regionCellArrays[i];

				int numComps = sourceArray->GetNumberOfComponents();
				std::vector<double> tuple(numComps);
				sourceArray->GetTuple(closestCellId, tuple.data());
				destArray->SetTuple(cid, tuple.data());
			}
		}

		// Attach all cell arrays to regionGrid
		for (auto& array : regionCellArrays) {
			array->Squeeze();
			grid->GetCellData()->AddArray(array);
		}
	}
}

void createRegionMappings()
{
	vtkDataArray* regColorIds = DATASET->GetCellData()->GetArray("RegionIds");
	// Find the number of cells and the points of the largest region

	std::unordered_map<int, std::map<vtkIdType, vtkIdType>> globalToLocalPointMap;
	std::unordered_map<int, vtkSmartPointer<vtkPoints>> regionPoints;
	std::unordered_map<int, vtkSmartPointer<vtkCellArray>> regionCells;

	vtkIdList* pointIds = vtkIdList::New();

	for (vtkIdType cellId = 0; cellId < DATASET->GetNumberOfCells(); ++cellId) {
		int regionId = regColorIds->GetTuple1(cellId);
		DATASET->GetCellPoints(cellId, pointIds);

		// Initialize structures for this region if first encountered
		if (regionsMap.find(regionId) == regionsMap.end()) {
			regionsMap[regionId] = vtkSmartPointer<vtkUnstructuredGrid>::New();
			regionPoints[regionId] = vtkSmartPointer<vtkPoints>::New();
			regionCells[regionId] = vtkSmartPointer<vtkCellArray>::New();
		}

		std::vector<vtkIdType> localPtIds(pointIds->GetNumberOfIds());

		for (vtkIdType j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			vtkIdType globalId = pointIds->GetId(j);

			if (globalToLocalPointMap[regionId].find(globalId) == globalToLocalPointMap[regionId].end()) {
				vtkIdType newId = regionPoints[regionId]->InsertNextPoint(DATASET->GetPoint(globalId));
				globalToLocalPointMap[regionId][globalId] = newId;
			}

			localPtIds[j] = globalToLocalPointMap[regionId][globalId];
		}

		regionCells[regionId]->InsertNextCell(static_cast<vtkIdType>(localPtIds.size()), localPtIds.data());
	}

	cout << "Computing point data..." << endl;
	int i = 0;
	// Assemble final vtkUnstructuredGrid objects
	for (auto& pair : regionsMap) {
		int regionId = pair.first;
		if (i % 1000 == 0) {
			cout << i << " " << regionsMap.size() << endl;
		}
		i++;
		vtkUnstructuredGrid* grid = pair.second;
		grid->SetPoints(regionPoints[regionId]);
		grid->SetCells(DATASET->GetCellType(0), regionCells[regionId]);

		// Interpolate point data from DATASET to grid
		vtkSmartPointer<vtkPointInterpolator> interpolator = vtkSmartPointer<vtkPointInterpolator>::New();
		interpolator->SetInputData(grid);     // Region grid with only geometry
		interpolator->SetSourceData(DATASET); // Dataset with original point data
		interpolator->SetLocator(PointLocator);
		// Use nearest-neighbor interpolation
		vtkSmartPointer<vtkVoronoiKernel> voronoiKernel = vtkSmartPointer<vtkVoronoiKernel>::New();
		interpolator->SetKernel(voronoiKernel);
		interpolator->SetNullPointsStrategyToClosestPoint(); // Use closest point if no exact match
		interpolator->Update();
		grid->DeepCopy(interpolator->GetOutput());
		grid->Squeeze();
	}

	pointIds->Delete();
	createCellData(regionsMap);
}

void createSegmentMappings()
{
	vtkDataArray* segColorIds = DATASET->GetCellData()->GetArray("SegmentIds");
	// Find the number of cells and the points of the largest region

	std::unordered_map<int, std::map<vtkIdType, vtkIdType>> globalToLocalPointMap;
	std::unordered_map<int, vtkSmartPointer<vtkPoints>> regionPoints;
	std::unordered_map<int, vtkSmartPointer<vtkCellArray>> regionCells;

	vtkIdList* pointIds = vtkIdList::New();

	for (vtkIdType cellId = 0; cellId < DATASET->GetNumberOfCells(); ++cellId) {
		int regionId = segColorIds->GetTuple1(cellId);
		DATASET->GetCellPoints(cellId, pointIds);

		// Initialize structures for this region if first encountered
		if (segmentsMap.find(regionId) == segmentsMap.end()) {
			segmentsMap[regionId] = vtkSmartPointer<vtkUnstructuredGrid>::New();
			regionPoints[regionId] = vtkSmartPointer<vtkPoints>::New();
			regionCells[regionId] = vtkSmartPointer<vtkCellArray>::New();
		}

		std::vector<vtkIdType> localPtIds(pointIds->GetNumberOfIds());

		for (vtkIdType j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			vtkIdType globalId = pointIds->GetId(j);

			if (globalToLocalPointMap[regionId].find(globalId) == globalToLocalPointMap[regionId].end()) {
				vtkIdType newId = regionPoints[regionId]->InsertNextPoint(DATASET->GetPoint(globalId));
				globalToLocalPointMap[regionId][globalId] = newId;
			}

			localPtIds[j] = globalToLocalPointMap[regionId][globalId];
		}

		regionCells[regionId]->InsertNextCell(static_cast<vtkIdType>(localPtIds.size()), localPtIds.data());
	}

	cout << "Computing point data..." << endl;
	int i = 0;
	// Assemble final vtkUnstructuredGrid objects
	for (auto& pair : segmentsMap) {
		int regionId = pair.first;
		if (i % 1000 == 0) {
			cout << i << " " << segmentsMap.size() << endl;
		}
		i++;
		vtkUnstructuredGrid* grid = pair.second;
		grid->SetPoints(regionPoints[regionId]);
		grid->SetCells(DATASET->GetCellType(0), regionCells[regionId]);

		// Interpolate point data from DATASET to grid
		vtkSmartPointer<vtkPointInterpolator> interpolator = vtkSmartPointer<vtkPointInterpolator>::New();
		interpolator->SetInputData(grid);     // Region grid with only geometry
		interpolator->SetSourceData(DATASET); // Dataset with original point data
		interpolator->SetLocator(PointLocator);
		// Use nearest-neighbor interpolation
		vtkSmartPointer<vtkVoronoiKernel> voronoiKernel = vtkSmartPointer<vtkVoronoiKernel>::New();
		interpolator->SetKernel(voronoiKernel);
		interpolator->SetNullPointsStrategyToClosestPoint(); // Use closest point if no exact match
		interpolator->Update();
		grid->DeepCopy(interpolator->GetOutput());
		grid->Squeeze();
	}

	pointIds->Delete();
	createCellData(segmentsMap);
}

void createContourMappings()
{
	vtkDataArray* regColorIds = ISOSURFACES->GetCellData()->GetArray("SurfaceRegionIds");
	// Find the number of cells and the points of the largest contour

	std::unordered_map<int, std::map<vtkIdType, vtkIdType>> globalToLocalPointMap;
	std::unordered_map<int, vtkSmartPointer<vtkPoints>> contourPoints;
	std::unordered_map<int, vtkSmartPointer<vtkCellArray>> contourCells;

	vtkIdList* pointIds = vtkIdList::New();

	for (vtkIdType cellId = 0; cellId < ISOSURFACES->GetNumberOfCells(); ++cellId) {
		int contourId = regColorIds->GetTuple1(cellId);
		ISOSURFACES->GetCellPoints(cellId, pointIds);

		// Initialize structures for this region if first encountered
		if (isosurfacesMap.find(contourId) == isosurfacesMap.end()) {
			isosurfacesMap[contourId] = vtkSmartPointer<vtkPolyData>::New();
			contourPoints[contourId] = vtkSmartPointer<vtkPoints>::New();
			contourCells[contourId] = vtkSmartPointer<vtkCellArray>::New();
		}

		std::vector<vtkIdType> localPtIds(pointIds->GetNumberOfIds());

		for (vtkIdType j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			vtkIdType globalId = pointIds->GetId(j);

			if (globalToLocalPointMap[contourId].find(globalId) == globalToLocalPointMap[contourId].end()) {
				vtkIdType newId = contourPoints[contourId]->InsertNextPoint(ISOSURFACES->GetPoint(globalId));
				globalToLocalPointMap[contourId][globalId] = newId;
			}

			localPtIds[j] = globalToLocalPointMap[contourId][globalId];
		}

		contourCells[contourId]->InsertNextCell(static_cast<vtkIdType>(localPtIds.size()), localPtIds.data());
	}

	cout << "Computing point data..." << endl;
	int i = 0;
	// Assemble final vtkUnstructuredGrid objects
	for (auto& pair : isosurfacesMap) {
		int regionId = pair.first;
		if (i % 1000 == 0) {
			cout << i << " " << isosurfacesMap.size() << endl;
		}
		i++;
		vtkPolyData* grid = pair.second;
		grid->SetPoints(contourPoints[regionId]);
		grid->SetPolys(contourCells[regionId]);
		grid->Squeeze();
	}
	pointIds->Delete();

}

void addVortexIdArray(vtkDataSet* dataset, vtkIdType& id)
{
	vtkSmartPointer<vtkIntArray> vortexIdArray = vtkSmartPointer<vtkIntArray>::New();
	vortexIdArray->SetNumberOfComponents(1);
	vortexIdArray->SetNumberOfTuples(dataset->GetNumberOfCells());
	vortexIdArray->SetName("VortexIds");
	for (int i = 0; i < vortexIdArray->GetNumberOfTuples(); i++)
	{
		vortexIdArray->SetTuple1(i, id);
	}
	dataset->GetCellData()->AddArray(vortexIdArray);
}

int main(int argc, char* argv[])
{
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	if (argc < 2)
	{
		std::cout << "Please specify the following arguments." << endl;
		std::cout << "1: input split regions file name." << endl;
		return EXIT_FAILURE;
	}
	else if (argc == 3) {
		CONSISTENCY = std::stof(argv[2]);
	}

	std::string temp = argv[1];

	// std::string datafile = "./skeleton_extraction/fortSkeleton.vtk";
	temp = argv[1];
	vtkSmartPointer<vtkDataSetReader> dataReader = vtkSmartPointer<vtkDataSetReader>::New();
	dataReader->SetFileName(temp.c_str());
	dataReader->Update();
	DATASET = dataReader->GetOutput();

	vtkPointSet* ps = vtkPointSet::SafeDownCast(DATASET);
	ps->BuildCellLocator();
	ps->BuildPointLocator();
	ps->EditableOff();

	cout << "Building point locator..." << endl;
	PointLocator = ps->GetPointLocator();
	cout << "Building cell locator..." << endl;
	CellLocator = ps->GetCellLocator();

	std::string name = "RegionIds";// +std::to_string(maxLevel);
	DATASET->GetCellData()->SetActiveScalars(name.c_str());
	createRegionMappings();
	// createSegmentMappings();

	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_Isosurfaces.vtk";
	vtkSmartPointer<vtkDataSetReader> surfaceReader = vtkSmartPointer<vtkDataSetReader>::New();
	surfaceReader->SetFileName(temp.c_str());
	surfaceReader->Update();
	ISOSURFACES = surfaceReader->GetOutput();

	vtkPointSet* isosurfaces_ps = vtkPointSet::SafeDownCast(ISOSURFACES);
	isosurfaces_ps->BuildCellLocator();
	isosurfaces_ps->EditableOff();

	cout << "Building surface cell locator..." << endl;
	isosurfacesCellLocator = isosurfaces_ps->GetCellLocator();
	createContourMappings();

	double lenLimit = DATASET->GetLength();
	std::vector<std::pair<vtkIdType, double>> regionDatasets;
	for (std::pair<vtkIdType, vtkSmartPointer<vtkUnstructuredGrid>> pair : regionsMap)
	{
		// Store this region into a vector for later retrival
		if (inspectRegion(pair.second) == false) {
			continue;
		}

		if (pair.second->GetLength() < lenLimit)
		{
			lenLimit = pair.second->GetLength();
		}
		regionDatasets.push_back(std::make_pair(pair.first, pair.second->GetBounds()[5]));
	}

	sort(regionDatasets.begin(), regionDatasets.end(), [](const std::pair<vtkIdType, double>& a,
		const std::pair<vtkIdType, double>& b) {
			return a.second > b.second;
		});
	// cout << regionDatasets[0].second->GetBounds()[5] << " " << regionDatasets[regionDatasets.size() - 1].second->GetBounds()[5] << " " << lenLimit << endl;
	double zMin = DATASET->GetBounds()[4];

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vtkSmartPointer<vtkAppendFilter> appendFilter = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter2 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter2->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter3 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter3->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter4 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter4->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter5 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter5->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter6 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter6->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter7 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter6->MergePointsOff();
	vtkSmartPointer<vtkAppendFilter> appendFilter8 = vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter6->MergePointsOff();
	std::chrono::steady_clock::time_point appendTimeBegin;
	std::chrono::steady_clock::time_point appendTimeEnd;

	appendTimeBegin = std::chrono::steady_clock::now();
	// First pass for type 1
	int i = 0;
	for (std::pair<vtkIdType, double>& pair : regionDatasets) {
		cout << i++ << "/" << regionDatasets.size() << " ";
		vtkSmartPointer<vtkPolyData> skeleton_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> vortex_lines_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_skel_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkUnstructuredGrid> regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkUnstructuredGrid> complete_regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkPolyData> isosurface_data = vtkSmartPointer<vtkPolyData>::New();
		// vtkNew<vtkPolyData> skeleton_data, surface_data, complete_surface_data, vortex_lines_data;
		// vtkNew<vtkUnstructuredGrid> regions_data, complete_regions_data;
		vtkSmartPointer<vtkUnstructuredGrid> regionDataset = regionsMap[pair.first];
		vtkIdType id = pair.first;
		// if (id != 5669) { continue;  }
		// Skip segments that have already been processed or overlap with the boundary.
		if (procesdRegions.count(id) > 0 || id == -1 || abs(regionDataset->GetBounds()[4] - zMin) <= lenLimit
			|| visitedRegions.count(id) > 0)
		{
			continue;
		}

		bool isHairpin = findAndJoinRegions(skeleton_data, surface_data, complete_surface_data, vortex_lines_data, regions_data,
			complete_regions_data, complete_skel_data, isosurface_data, id, 1, i);
		cout << isHairpin << endl;
		if (skeleton_data->GetNumberOfCells() == 0 || surface_data->GetNumberOfCells() == 0 || isHairpin == false)
		{
			continue;
		}

		addVortexIdArray(skeleton_data, id);
		appendFilter->AddInputData(skeleton_data);
		addVortexIdArray(surface_data, id);
		appendFilter2->AddInputData(surface_data);
		addVortexIdArray(complete_surface_data, id);
		appendFilter3->AddInputData(complete_surface_data);
		addVortexIdArray(regions_data, id);
		appendFilter4->AddInputData(regions_data);
		addVortexIdArray(complete_regions_data, id);
		appendFilter5->AddInputData(complete_regions_data);
		addVortexIdArray(vortex_lines_data, id);
		appendFilter6->AddInputData(vortex_lines_data);
		addVortexIdArray(complete_skel_data, id);
		appendFilter7->AddInputData(complete_skel_data);
		addVortexIdArray(isosurface_data, id);
		appendFilter8->AddInputData(isosurface_data);
	}
	appendTimeEnd = std::chrono::steady_clock::now();
	float appendTime1 = std::chrono::duration_cast<std::chrono::milliseconds>(appendTimeEnd - appendTimeBegin).count();

	regionStrings.clear();
	appendTimeBegin = std::chrono::steady_clock::now();
	// Second pass for type 4
	i = 0;
	for (std::pair<vtkIdType, double>& pair : regionDatasets) {
		cout << i++ << "/" << regionDatasets.size() << " ";
		vtkSmartPointer<vtkPolyData> skeleton_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> vortex_lines_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_skel_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkUnstructuredGrid> regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkUnstructuredGrid> complete_regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkPolyData> isosurface_data = vtkSmartPointer<vtkPolyData>::New();
		// vtkNew<vtkPolyData> skeleton_data, surface_data, complete_surface_data, vortex_lines_data;
		// vtkNew<vtkUnstructuredGrid> regions_data, complete_regions_data;
		vtkSmartPointer<vtkUnstructuredGrid> regionDataset = regionsMap[pair.first];
		vtkIdType id = pair.first;

		// Skip segments that have already been processed or overlap with the boundary.
		if (procesdRegions.count(id) > 0 || id == -1 || abs(regionDataset->GetBounds()[4] - zMin) <= lenLimit || visitedRegions.count(id) > 0) {
			continue;
		}
		// cout << "Reg Num: " << id << " ";
		bool isHairpin = findAndJoinRegions(skeleton_data, surface_data, complete_surface_data, vortex_lines_data, regions_data,
			complete_regions_data, complete_skel_data, isosurface_data, id, 4, i);
		cout << isHairpin << endl;
		if (skeleton_data->GetNumberOfCells() == 0 || surface_data->GetNumberOfCells() == 0 || isHairpin == false)
		{
			continue;
		}

		addVortexIdArray(skeleton_data, id);
		appendFilter->AddInputData(skeleton_data);
		addVortexIdArray(surface_data, id);
		appendFilter2->AddInputData(surface_data);
		addVortexIdArray(complete_surface_data, id);
		appendFilter3->AddInputData(complete_surface_data);
		addVortexIdArray(regions_data, id);
		appendFilter4->AddInputData(regions_data);
		addVortexIdArray(complete_regions_data, id);
		appendFilter5->AddInputData(complete_regions_data);
		addVortexIdArray(vortex_lines_data, id);
		appendFilter6->AddInputData(vortex_lines_data);
		addVortexIdArray(complete_skel_data, id);
		appendFilter7->AddInputData(complete_skel_data);
		addVortexIdArray(isosurface_data, id);
		appendFilter8->AddInputData(isosurface_data);
	}
	appendTimeEnd = std::chrono::steady_clock::now();
	float appendTime2 = std::chrono::duration_cast<std::chrono::milliseconds>(appendTimeEnd - appendTimeBegin).count();

	regionStrings.clear();
	appendTimeBegin = std::chrono::steady_clock::now();
	// Third pass for type 2
	i = 0;
	for (std::pair<vtkIdType, double>& pair : regionDatasets) {
		cout << i++ << "/" << regionDatasets.size() << " ";
		vtkSmartPointer<vtkPolyData> skeleton_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> vortex_lines_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_skel_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkUnstructuredGrid> regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkUnstructuredGrid> complete_regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkPolyData> isosurface_data = vtkSmartPointer<vtkPolyData>::New();
		// vtkNew<vtkPolyData> skeleton_data, surface_data, complete_surface_data, vortex_lines_data;
		// vtkNew<vtkUnstructuredGrid> regions_data, complete_regions_data;
		vtkSmartPointer<vtkUnstructuredGrid> regionDataset = regionsMap[pair.first];
		vtkIdType id = pair.first;

		// Skip segments that have already been processed or overlap with the boundary.
		if (procesdRegions.count(id) > 0 || id == -1 || abs(regionDataset->GetBounds()[4] - zMin) <= lenLimit || visitedRegions.count(id) > 0) {
			continue;
		}
		// cout << "Reg Num: " << id << " ";
		bool isHairpin = findAndJoinRegions(skeleton_data, surface_data, complete_surface_data, vortex_lines_data, regions_data,
			complete_regions_data, complete_skel_data, isosurface_data, id, 2, i);
		cout << isHairpin << endl;
		if (skeleton_data->GetNumberOfCells() == 0 || surface_data->GetNumberOfCells() == 0 || isHairpin == false)
		{
			continue;
		}

		addVortexIdArray(skeleton_data, id);
		appendFilter->AddInputData(skeleton_data);
		addVortexIdArray(surface_data, id);
		appendFilter2->AddInputData(surface_data);
		addVortexIdArray(complete_surface_data, id);
		appendFilter3->AddInputData(complete_surface_data);
		addVortexIdArray(regions_data, id);
		appendFilter4->AddInputData(regions_data);
		addVortexIdArray(complete_regions_data, id);
		appendFilter5->AddInputData(complete_regions_data);
		addVortexIdArray(vortex_lines_data, id);
		appendFilter6->AddInputData(vortex_lines_data);
		addVortexIdArray(complete_skel_data, id);
		appendFilter7->AddInputData(complete_skel_data);
		addVortexIdArray(isosurface_data, id);
		appendFilter8->AddInputData(isosurface_data);
	}
	appendTimeEnd = std::chrono::steady_clock::now();
	float appendTime3 = std::chrono::duration_cast<std::chrono::milliseconds>(appendTimeEnd - appendTimeBegin).count();

	regionStrings.clear();
	appendTimeBegin = std::chrono::steady_clock::now();
	// 5th pass for type 5
	i = 0;
	for (std::pair<vtkIdType, double>& pair : regionDatasets) {
		cout << i++ << "/" << regionDatasets.size() << " ";
		vtkSmartPointer<vtkPolyData> skeleton_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_surface_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> vortex_lines_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkPolyData> complete_skel_data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkUnstructuredGrid> regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkUnstructuredGrid> complete_regions_data = vtkSmartPointer<vtkUnstructuredGrid>::New();
		vtkSmartPointer<vtkPolyData> isosurface_data = vtkSmartPointer<vtkPolyData>::New();
		// vtkNew<vtkPolyData> skeleton_data, surface_data, complete_surface_data, vortex_lines_data;
		// vtkNew<vtkUnstructuredGrid> regions_data, complete_regions_data;
		vtkSmartPointer<vtkUnstructuredGrid> regionDataset = regionsMap[pair.first];
		vtkIdType id = pair.first;

		// Skip segments that have already been processed or overlap with the boundary.
		if (procesdRegions.count(id) > 0 || id == -1 || abs(regionDataset->GetBounds()[4] - zMin) <= lenLimit || visitedRegions.count(id) > 0) {
			continue;
		}
		// cout << "Reg Num: " << id << " ";
		bool isHairpin = findAndJoinRegions(skeleton_data, surface_data, complete_surface_data, vortex_lines_data, regions_data,
			complete_regions_data, complete_skel_data, isosurface_data, id, 5, i);
		cout << isHairpin << endl;
		if (skeleton_data->GetNumberOfCells() == 0 || surface_data->GetNumberOfCells() == 0 || isHairpin == false)
		{
			continue;
		}

		addVortexIdArray(skeleton_data, id);
		appendFilter->AddInputData(skeleton_data);
		addVortexIdArray(surface_data, id);
		appendFilter2->AddInputData(surface_data);
		addVortexIdArray(complete_surface_data, id);
		appendFilter3->AddInputData(complete_surface_data);
		addVortexIdArray(regions_data, id);
		appendFilter4->AddInputData(regions_data);
		addVortexIdArray(complete_regions_data, id);
		appendFilter5->AddInputData(complete_regions_data);
		addVortexIdArray(vortex_lines_data, id);
		appendFilter6->AddInputData(vortex_lines_data);
		addVortexIdArray(complete_skel_data, id);
		appendFilter7->AddInputData(complete_skel_data);
		addVortexIdArray(isosurface_data, id);
		appendFilter8->AddInputData(isosurface_data);
	}
	appendTimeEnd = std::chrono::steady_clock::now();
	float appendTime5 = std::chrono::duration_cast<std::chrono::milliseconds>(appendTimeEnd - appendTimeBegin).count();

	appendFilter->Update();
	appendFilter2->Update();
	appendFilter3->Update();
	appendFilter4->Update();
	appendFilter5->Update();
	appendFilter6->Update();
	appendFilter7->Update();
	appendFilter8->Update();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
	cout << "Time difference streamTracer = " << streamTracerTime / 1000 << "[s]" << std::endl;
	cout << "Time difference findCandidates = " << findCandidatesTime / 1000 << "[s]" << std::endl;
	cout << "Time difference surfaceExtraction = " << surfaceExtractionTime / 1000 << "[s]" << std::endl;
	cout << "Time difference skelExtraction = " << skelExtractionTime / 1000 << "[s]" << std::endl;
	cout << "Time difference skelFindTime = " << skelFindTime / 1000 << "[s]" << std::endl;
	cout << "================SkelFindTimeBreakdown===================" << endl;
	cout << "Time difference interpolatorTime = " << interpolatorTime / 1000 << "[s]" << std::endl;
	cout << "Time difference cleanTime = " << cleanTime / 1000 << "[s]" << std::endl;
	cout << "Time difference smoothFilterTime = " << smoothFilterTime / 1000 << "[s]" << std::endl;
	cout << "===================================" << endl;
	cout << "Time difference postProcessingTime = " << postProcessingTime / 1000 << "[s]" << std::endl;
	cout << "Time difference appendTime = " << appendTime1 / 1000 << "[s]" << std::endl;
	cout << "Time difference appendTime = " << appendTime2 / 1000 << "[s]" << std::endl;
	cout << "Time difference appendTime = " << appendTime3 / 1000 << "[s]" << std::endl;
	// cout << "Time difference appendTime = " << appendTime4 / 1000 << "[s]" << std::endl;
	cout << "Time difference appendTime = " << appendTime5 / 1000 << "[s]" << std::endl;
	// cout << "Time difference appendTime = " << appendTime6 / 1000 << "[s]" << std::endl;
	// cout << "Time difference appendTime = " << appendTime7 / 1000 << "[s]" << std::endl;

	vtkNew<vtkDataSetWriter> writer;
	writer->SetInputData(appendFilter->GetOutput());
	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_hairpinSkeleton.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter2->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter2->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter2->GetOutput());
	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_hairpinSurface.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter3->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter3->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter3->GetOutput());
	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_fullSurface.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter4->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter4->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter4->GetOutput());
	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_hairpinRegion.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter5->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter5->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter5->GetOutput());
	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_fullRegion.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter6->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter6->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter6->GetOutput());

	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_vortexLines.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter7->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter7->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter7->GetOutput());

	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_fullSkeleton.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	appendFilter8->GetOutput()->GetCellData()->RemoveArray("RegionId");
	appendFilter8->GetOutput()->GetPointData()->RemoveArray("RegionId");
	writer->SetInputData(appendFilter8->GetOutput());

	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_hairpinIsosurface.vtk";
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	return EXIT_SUCCESS;
}