// In this version, we get contour tree-based segmentation only once for each separate region and perform layering on all minima.

#include <vtkAppendDataSets.h>
#include <vtkConnectivityFilter.h>
#include <vtkContourFilter.h>
#include <vtkCellCenters.h>
#include <vtkDataSet.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkDataSetMapper.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkGeometryFilter.h>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkProperty.h>
#include <vtkPointInterpolator.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSMPTools.h>
#include <vtkStaticCellLocator.h>
#include <vtkStaticPointLocator.h>
#include <vtkThreshold.h>
#include <vtkVoronoiKernel.h>

#include <ttkPersistenceCurve.h>
#include <ttkPersistenceDiagram.h>
#include <ttkTopologicalSimplification.h>
#include <ttkMergeTree.h>
#include <ttkManifoldCheck.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <random>

std::unordered_map<int, vtkSmartPointer<vtkIdList>> TETRA_MAPPINGS;
std::ofstream logFile;
int MAX_REG_ID;
int DATASIZE = 0;

struct Region
{
	int RegionId;
	vtkSmartPointer<vtkUnstructuredGrid> RegionData;

	Region(int regionId, vtkUnstructuredGrid* regionData)
	{
		this->RegionId = regionId;
		this->RegionData = vtkSmartPointer<vtkUnstructuredGrid>::New();
		this->RegionData->DeepCopy(regionData);
	}

	// Destructor
	~Region()
	{
		// Clean up resources
		this->RegionData = nullptr; // This line is optional if RegionData is a smart pointer
	}
};
std::unordered_map<int, vtkSmartPointer<vtkUnstructuredGrid>> REGIONS;

bool compareCount(const std::pair<int, int>& a, const std::pair<int, int>& b) {
	return a.second > b.second; // Sort in descending order of count
}

void getDisconnectedRegions(vtkDataSet* dataset, int& numRegions) {
	vtkSmartPointer<vtkConnectivityFilter> connFilter = vtkSmartPointer<vtkConnectivityFilter>::New();
	connFilter->SetInputData(dataset);
	connFilter->SetRegionIdAssignmentMode(vtkConnectivityFilter::CELL_COUNT_DESCENDING);
	connFilter->SetExtractionModeToAllRegions();
	connFilter->ColorRegionsOn();
	connFilter->Update();
	vtkSmartPointer<vtkIntArray> RegionIdArr = vtkSmartPointer<vtkIntArray>::New();
	RegionIdArr->DeepCopy(connFilter->GetOutput()->GetCellData()->GetArray("RegionId"));
	numRegions = connFilter->GetNumberOfExtractedRegions();
	connFilter->GetOutput()->ReleaseData();
	dataset->GetCellData()->AddArray(RegionIdArr);
}

void performManifoldCheck(vtkDataSet* dataset) {
	vtkSmartPointer<ttkManifoldCheck> manCheck = vtkSmartPointer<ttkManifoldCheck>::New();
	manCheck->SetInputData(dataset);
	manCheck->Update();
	vtkSmartPointer<vtkIntArray> VertexLinkComponentNumberArr = vtkSmartPointer<vtkIntArray>::New();
	VertexLinkComponentNumberArr->DeepCopy(manCheck->GetOutput()->GetCellData()->GetArray("VertexLinkComponentNumber"));
	manCheck->GetOutput()->ReleaseData();
	dataset->GetCellData()->AddArray(VertexLinkComponentNumberArr);
}

void performThreshold(vtkDataSet* dataset, std::string arrayName) {
	vtkSmartPointer<vtkThreshold> manThreshold = vtkSmartPointer<vtkThreshold>::New();
	manThreshold->SetInputData(dataset);
	manThreshold->SetInputArrayToProcess(0, 0, 0,
		vtkDataObject::FIELD_ASSOCIATION_CELLS, arrayName.c_str());
	manThreshold->SetLowerThreshold(1);
	manThreshold->SetUpperThreshold(1);
	manThreshold->Update();
	manThreshold->GetOutput()->GetCellData()->RemoveArray(arrayName.c_str());
	dataset->DeepCopy(manThreshold->GetOutput());
}

void createRegionMappings(vtkDataSet* DATASET)
{
	vtkDataArray* regColorIds = DATASET->GetCellData()->GetArray("RegionId");
	// Find the number of cells and the points of the largest region

	std::unordered_map<int, std::map<int, int>> globalToLocalPointMap;
	std::unordered_map<int, vtkSmartPointer<vtkPoints>> regionPoints;
	std::unordered_map<int, vtkSmartPointer<vtkCellArray>> regionCells;

	vtkIdList* pointIds = vtkIdList::New();

	for (int cellId = 0; cellId < DATASET->GetNumberOfCells(); ++cellId) {
		int regionId = regColorIds->GetTuple1(cellId);
		DATASET->GetCellPoints(cellId, pointIds);

		// Initialize structures for this region if first encountered
		if (REGIONS.find(regionId) == REGIONS.end()) {
			REGIONS[regionId] = vtkSmartPointer<vtkUnstructuredGrid>::New();
			regionPoints[regionId] = vtkSmartPointer<vtkPoints>::New();
			regionCells[regionId] = vtkSmartPointer<vtkCellArray>::New();
		}

		std::vector<vtkIdType> localPtIds(pointIds->GetNumberOfIds());

		for (int j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			int globalId = pointIds->GetId(j);

			if (globalToLocalPointMap[regionId].find(globalId) == globalToLocalPointMap[regionId].end()) {
				int newId = regionPoints[regionId]->InsertNextPoint(DATASET->GetPoint(globalId));
				globalToLocalPointMap[regionId][globalId] = newId;
			}

			localPtIds[j] = globalToLocalPointMap[regionId][globalId];
		}

		regionCells[regionId]->InsertNextCell(static_cast<int>(localPtIds.size()), localPtIds.data());
	}

	vtkSmartPointer<vtkStaticPointLocator> locator = vtkSmartPointer<vtkStaticPointLocator>::New();
	locator->SetDataSet(DATASET);
	locator->BuildLocator();

	int i = 0;
	// Assemble final vtkUnstructuredGrid objects
	for (auto& pair : REGIONS) {
		int regionId = pair.first;
		cout << "Preparing region: " << i++ << " Total: " << REGIONS.size() << endl;
		vtkUnstructuredGrid* grid = pair.second;
		grid->SetPoints(regionPoints[regionId]);
		grid->SetCells(VTK_TETRA, regionCells[regionId]); // You can replace with actual type if known

		// Interpolate point data from DATASET to grid
		vtkSmartPointer<vtkPointInterpolator> interpolator = vtkSmartPointer<vtkPointInterpolator>::New();
		interpolator->SetInputData(grid);     // Region grid with only geometry
		interpolator->SetSourceData(DATASET); // Dataset with original point data
		interpolator->SetLocator(locator);
		// Use nearest-neighbor interpolation
		vtkSmartPointer<vtkVoronoiKernel> voronoiKernel = vtkSmartPointer<vtkVoronoiKernel>::New();
		interpolator->SetKernel(voronoiKernel);
		interpolator->SetNullPointsStrategyToClosestPoint(); // Use closest point if no exact match
		interpolator->Update();
		grid->DeepCopy(interpolator->GetOutput());
	}

	pointIds->Delete();
}

void getManifoldMultiRegion(vtkDataSet* dataset)
{
	int totalRegs = 0;
	cout << "Checking manifold.." << endl;
	performManifoldCheck(dataset);
	performThreshold(dataset, "VertexLinkComponentNumber");
	int numRegs;
	getDisconnectedRegions(dataset, numRegs);
	createRegionMappings(dataset);

	/*
	vtkSmartPointer<vtkAppendDataSets> append = vtkSmartPointer<vtkAppendDataSets>::New();
	for (auto& pair : REGIONS)
	{
		vtkSmartPointer<vtkUnstructuredGrid> temp = pair.second;
		temp->GetPointData()->RemoveArray("RegionId");
		temp->GetCellData()->RemoveArray("RegionId");
		append->AddInputData(temp);
	}
	append->Update();
	dataset->DeepCopy(append->GetOutput());
	append->GetOutput()->ReleaseData();

	vtkSmartPointer<vtkIntArray> GlobalCellIds = vtkSmartPointer<vtkIntArray>::New();
	GlobalCellIds->SetNumberOfComponents(1);
	GlobalCellIds->SetNumberOfTuples(dataset->GetNumberOfCells());
	GlobalCellIds->SetName("GlobalCellIds");
	for (int i = 0; i < dataset->GetNumberOfCells(); ++i)
	{
		GlobalCellIds->SetTuple1(i, i);
	}
	dataset->GetCellData()->AddArray(GlobalCellIds);

	vtkSmartPointer<vtkStaticCellLocator> CellLocator = vtkSmartPointer<vtkStaticCellLocator>::New();
	CellLocator->AutomaticOn();
	CellLocator->SetNumberOfCellsPerNode(1);
	CellLocator->SetDataSet(dataset);
	CellLocator->BuildLocator();

	// Create a reusable cell object for getting data from DATASET
	vtkSmartPointer<vtkGenericCell> tempCell = vtkSmartPointer<vtkGenericCell>::New();
	vtkCellData* inputCellData = dataset->GetCellData();
	int numCellArrays = inputCellData->GetNumberOfArrays();

	for (auto& pair : REGIONS) {
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
		for (int cid = 0; cid < grid->GetNumberOfCells(); ++cid) {
			double center[3] = { 0, 0, 0 };
			vtkCell* cell = grid->GetCell(cid);
			double pt[3];
			for (int i = 0; i < cell->GetNumberOfPoints(); ++i) {
				grid->GetPoint(cell->GetPointId(i), pt);
				center[0] += pt[0];
				center[1] += pt[1];
				center[2] += pt[2];
			}
			center[0] /= cell->GetNumberOfPoints();
			center[1] /= cell->GetNumberOfPoints();
			center[2] /= cell->GetNumberOfPoints();

			int closestCellId = CellLocator->FindCell(center);
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
	*/
}

int getCellNeighbors(vtkDataSet* dataset, vtkIdList* uniqueIds, int tetraId)
{
	vtkDataArray* originalCellIds = dataset->GetCellData()->GetArray("OriginalCellIds");
	int originalId = originalCellIds->GetTuple1(tetraId);
	vtkNew<vtkIdList> cellPoints;
	// First find out the 6 tetra that have the same original id
	dataset->GetCellPoints(tetraId, cellPoints);
	for (int i = 0; i < cellPoints->GetNumberOfIds(); i++)
	{
		vtkNew<vtkIdList> idList;
		idList->InsertNextId(cellPoints->GetId(i));

		// Get the neighbors of the cell.
		vtkNew<vtkIdList> neighborCellIds;
		dataset->GetCellNeighbors(tetraId, idList, neighborCellIds);

		for (int j = 0; j < neighborCellIds->GetNumberOfIds(); j++)
		{
			int neighborCellId = neighborCellIds->GetId(j);
			int thisId = originalCellIds->GetTuple1(neighborCellId);
			if (thisId == originalId)
			{
				uniqueIds->InsertUniqueId(neighborCellId);
			}
		}
	}
	return originalId;
}

void getTetraMappings(vtkDataSet* dataset)
{
	TETRA_MAPPINGS.clear();
	vtkNew<vtkIdList> visited;
	int numIter = 0;
	int origId;
	for (int tetraId = 0; tetraId < dataset->GetNumberOfCells(); tetraId++)
	{
		visited->Reset();
		vtkNew<vtkIdList> uniqueIds;
		uniqueIds->InsertUniqueId(tetraId);
		origId = getCellNeighbors(dataset, uniqueIds, tetraId);
		while (uniqueIds->GetNumberOfIds() < 6)
		{
			numIter += 1;
			for (int i = 0; i < uniqueIds->GetNumberOfIds(); i++)
			{
				int newTetraId = uniqueIds->GetId(i);
				if (visited->FindIdLocation(newTetraId) != -1)
				{
					continue;
				}
				getCellNeighbors(dataset, uniqueIds, newTetraId);
				visited->InsertNextId(newTetraId);
			}

			if (numIter > 26)	// Number of neighbors of a certain voxel
			{
				break;
			}
		}

		TETRA_MAPPINGS[origId] = uniqueIds;
	}
}

void updateTetraIds(vtkDataSet* dataset, vtkDataArray* segmentationCellIdArr,
	std::vector<std::pair<int, int>>& changedColorIds)
{
	vtkDataArray* originalCellIds = dataset->GetCellData()->GetArray("OriginalCellIds");
	vtkDataArray* globalCellIds = dataset->GetCellData()->GetArray("GlobalCellIds");

	vtkSmartPointer<vtkIdList> uniqueIds;
	vtkNew<vtkIdList> visited;
	int globalCellId, colorId, uniqueId, origId;
	std::vector <std::pair<int, int>> colorIdsCount;
	bool found;
	for (std::pair<int, int>& pair : changedColorIds)
	{
		colorIdsCount.clear();
		int tetraId = pair.first;
		if (visited->FindIdLocation(tetraId) != -1)
		{
			continue;
		}
		origId = originalCellIds->GetTuple1(tetraId);
		uniqueIds = TETRA_MAPPINGS[origId];

		// Go over each unique id and find the count of their color ids
		for (int i = 0; i < uniqueIds->GetNumberOfIds(); i++)
		{
			uniqueId = uniqueIds->GetId(i);
			visited->InsertNextId(uniqueId);
			colorId = segmentationCellIdArr->GetTuple1(uniqueId);
			found = false;
			for (int j = 0; j < colorIdsCount.size(); j++)
			{
				if (colorIdsCount[j].first == colorId)
				{
					colorIdsCount[j].second++;
					found = true;
					break;
				}
			}

			if (!found)
			{
				std::pair<int, int> pair;
				pair.first = colorId;
				pair.second = 1;
				colorIdsCount.push_back(pair);
			}
		}

		if (colorIdsCount.size() == 1)
		{
			continue;
		}

		int maxColorId = -1;
		int maxColorIdCount = -1;
		for (std::pair<int, int>& pair : colorIdsCount)
		{
			if (pair.second > maxColorIdCount)
			{
				maxColorIdCount = pair.second;
				maxColorId = pair.first;
			}
		}

		if (maxColorId == -1)
		{
			continue;
		}
		// cout << maxColorId << " " << maxColorIdCount << endl;
		// logFile << maxColorId << " " << maxColorIdCount << endl;

		for (int i = 0; i < uniqueIds->GetNumberOfIds(); i++)
		{
			uniqueId = uniqueIds->GetId(i);
			colorId = segmentationCellIdArr->GetTuple1(uniqueId);
			if (colorId != maxColorId)
			{
				segmentationCellIdArr->SetTuple1(uniqueId, maxColorId);
			}
		}
	}
}

void updateTetraCellIds(vtkDataSet* dataset, vtkDataArray* segmentationCellIdArr)
{
	vtkDataArray* originalCellIds = dataset->GetCellData()->GetArray("OriginalCellIds");

	vtkSmartPointer<vtkIdList> uniqueIds;
	std::unordered_set<int> visited;
	int globalCellId, colorId, uniqueId, origId;
	std::unordered_map<int, int> colorIdsCount;
	bool found;
	for (int tetraId = 0; tetraId < dataset->GetNumberOfCells(); tetraId++)
	{
		colorIdsCount.clear();

		if (visited.count(tetraId) == 1)
		{
			continue;
		}

		origId = originalCellIds->GetTuple1(tetraId);
		uniqueIds = TETRA_MAPPINGS[origId];

		// Go over each unique id and find the count of their color ids
		for (int i = 0; i < uniqueIds->GetNumberOfIds(); i++)
		{
			uniqueId = uniqueIds->GetId(i);
			visited.insert(uniqueId);
			colorId = segmentationCellIdArr->GetTuple1(uniqueId);
			colorIdsCount[colorId]++;
		}

		if (colorIdsCount.size() <= 1)
		{
			continue;
		}

		int maxColorId = -1;
		int maxColorIdCount = -1;
		for (std::pair<int, int> pair : colorIdsCount)
		{
			if (pair.second > maxColorIdCount)
			{
				maxColorIdCount = pair.second;
				maxColorId = pair.first;
			}
		}

		for (int i = 0; i < uniqueIds->GetNumberOfIds(); i++)
		{
			uniqueId = uniqueIds->GetId(i);
			segmentationCellIdArr->SetTuple1(uniqueId, maxColorId);
		}
	}
}

void getImmediateNeighborsEdge(vtkDataSet* dataset, int id, vtkIdList* neighborCells)
{
	neighborCells->Reset();
	// vtkNew<vtkGenericCell> cell;
	// dataset->GetCell(id, cell);
	vtkCell* cell = dataset->GetCell(id);
	vtkNew<vtkIdList> neighborCellIds, pointCells;
	for (int edgeId = 0; edgeId < cell->GetNumberOfEdges(); edgeId++)
	{
		pointCells->Reset();
		neighborCellIds->Reset();
		vtkCell* edge = cell->GetEdge(edgeId);
		vtkIdList* edgePoints = edge->GetPointIds();
		for (int ptId = 0; ptId < edgePoints->GetNumberOfIds(); ptId++)
		{
			pointCells->InsertNextId(edgePoints->GetId(ptId));
		}
		dataset->GetCellNeighbors(id, pointCells, neighborCellIds);
		for (int j = 0; j < neighborCellIds->GetNumberOfIds(); j++)
		{
			neighborCells->InsertNextId(neighborCellIds->GetId(j));
		}
	}
	neighborCells->Sort();
}

void getImmediateNeighborsFace(vtkDataSet* dataset, int id, vtkIdList* neighborCells)
{
	neighborCells->Reset();
	// vtkNew<vtkGenericCell> cell;
	// dataset->GetCell(id, cell);
	vtkCell* cell = dataset->GetCell(id);
	vtkSmartPointer<vtkIdList> neighborCellIds = vtkSmartPointer<vtkIdList>::New();
	vtkSmartPointer<vtkIdList> pointCells = vtkSmartPointer<vtkIdList>::New();
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
		dataset->GetCellNeighbors(id, pointCells, neighborCellIds);
		for (int j = 0; j < neighborCellIds->GetNumberOfIds(); j++)
		{
			neighborCells->InsertNextId(neighborCellIds->GetId(j));
		}
	}
	neighborCells->Sort();
}

void getImmediateNeighborsEdgeMP(vtkDataSet* dataset, int id, vtkIdList* neighborCells)
{
	neighborCells->Reset();
	vtkNew<vtkGenericCell> cell;
	dataset->GetCell(id, cell);
	// vtkCell* cell = dataset->GetCell(id);
	vtkNew<vtkIdList> neighborCellIds, pointCells;
	for (int edgeId = 0; edgeId < cell->GetNumberOfEdges(); edgeId++)
	{
		pointCells->Reset();
		neighborCellIds->Reset();
		vtkCell* edge = cell->GetEdge(edgeId);
		vtkIdList* edgePoints = edge->GetPointIds();
		for (int ptId = 0; ptId < edgePoints->GetNumberOfIds(); ptId++)
		{
			pointCells->InsertNextId(edgePoints->GetId(ptId));
		}
		dataset->GetCellNeighbors(id, pointCells, neighborCellIds);
		for (int j = 0; j < neighborCellIds->GetNumberOfIds(); j++)
		{
			neighborCells->InsertNextId(neighborCellIds->GetId(j));
		}
	}
	neighborCells->Sort();
}

void getImmediateNeighbors(vtkDataSet* dataset, int id, vtkIdList* neighborCells)
{
	neighborCells->Reset();
	vtkNew<vtkIdList> neighborCellIds, cellPoints, pointCells;

	// Find ids of all the cell neighbors of the current cell
	dataset->GetCellPoints(id, cellPoints);

	for (int ptId = 0; ptId < cellPoints->GetNumberOfIds(); ptId++)
	{
		pointCells->Reset();
		neighborCellIds->Reset();
		// Add one of the edge points.
		pointCells->InsertNextId(cellPoints->GetId(ptId));

		if (ptId + 1 == cellPoints->GetNumberOfIds())
		{
			pointCells->InsertNextId(cellPoints->GetId(0));
		}
		else
		{
			pointCells->InsertNextId(cellPoints->GetId(ptId + 1));
		}

		dataset->GetCellNeighbors(id, pointCells, neighborCellIds);
		for (int j = 0; j < neighborCellIds->GetNumberOfIds(); j++)
		{
			neighborCells->InsertNextId(neighborCellIds->GetId(j));
		}
	}
	neighborCells->Sort();
}

void assignSegmentationCellIds(vtkDataSet* dataset)
{
	vtkNew<vtkIdList> cellPointIds;
	std::vector<std::pair<int, int>> cellPtsSegIds;
	vtkDataArray* SegmentationIds = dataset->GetPointData()->GetArray("SegmentationId");
	vtkNew<vtkIntArray> cellSegmentationIds;
	cellSegmentationIds->SetName("SegmentationCellId");
	cellSegmentationIds->SetNumberOfComponents(1);
	cellSegmentationIds->SetNumberOfTuples(dataset->GetNumberOfCells());
	for (int i = 0; i < dataset->GetNumberOfCells(); i++)
	{
		// for each cell count the majority of ids
		cellPtsSegIds.clear();
		dataset->GetCellPoints(i, cellPointIds);
		for (int j = 0; j < cellPointIds->GetNumberOfIds(); j++)
		{
			int ptId = cellPointIds->GetId(j);
			int segId = SegmentationIds->GetTuple1(ptId);
			// Check if the segId already exists in the vector, if Yes increment its count
			bool found = false;
			for (std::pair<int, int>& pair : cellPtsSegIds)
			{
				if (pair.first == segId)
				{
					pair.second++;
					found = true;
					break;
				}
			}

			if (found == false)
			{
				std::pair<int, int> pair(segId, 1);
				cellPtsSegIds.push_back(pair);
			}
		}

		int maxSegId, maxSegCount = -1;
		for (std::pair<int, int>& pair : cellPtsSegIds)
		{
			if (pair.second > maxSegCount)
			{
				maxSegId = pair.first;
				maxSegCount = pair.second;
			}
		}

		cellSegmentationIds->SetTuple1(i, maxSegId);
	}
	cellSegmentationIds->Modified();
	dataset->GetCellData()->AddArray(cellSegmentationIds);
	dataset->GetPointData()->RemoveArray(SegmentationIds->GetName());
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

	// If the number of regions is larger than the number of specified colors,
	// generate some random colors.
	// Note: If a Python version is written, it is probably best to use
	//       vtkMinimalStandardRandomSequence in it and here, to ensure
	//       that the random number generation is the same.

	if (numberOfRegions > 9)
	{
		std::mt19937 mt(4355412); // Standard mersenne_twister_engine
		std::uniform_real_distribution<double> distribution(.1, 0.5);
		for (auto i = 10; i < numberOfRegions; ++i)
		{
			lut->SetTableValue(i, distribution(mt), distribution(mt),
				distribution(mt), 1.0);
		}
	}

	lut->SetTableValue(numberOfRegions, colors->GetColor4d("DarkRed").GetData());
}

void performTopologicalSimplification(vtkDataSet* dataset,
	vtkDataSet* persistentPairs, std::string& scalarName) {
	// 6. simplifying the input data to remove non-persistent pairs
	vtkSmartPointer<ttkTopologicalSimplification> topologicalSimplification =
		vtkSmartPointer<ttkTopologicalSimplification>::New();
	topologicalSimplification->SetInputData(0, dataset);
	topologicalSimplification->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, scalarName.c_str());
	topologicalSimplification->SetInputData(1, persistentPairs);
	topologicalSimplification->Update();

	vtkSmartPointer<vtkFloatArray> scalarArr = vtkSmartPointer<vtkFloatArray>::New();
	scalarArr->DeepCopy(topologicalSimplification->GetOutput()->GetPointData()->GetArray(scalarName.c_str()));
	topologicalSimplification->GetOutput()->ReleaseData();
	dataset->GetPointData()->AddArray(scalarArr);
}

void getContourTree(vtkDataSet* dataset, vtkDataSet* treeArcs, vtkDataSet* treeNodes,
	std::string& scalarName) {
	vtkSmartPointer<ttkMergeTree> contourTree = vtkSmartPointer<ttkMergeTree>::New();
	contourTree->SetInputData(dataset);
	contourTree->SetTreeType(0);
	contourTree->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, scalarName.c_str());
	contourTree->Update();

	vtkSmartPointer<vtkIntArray> SegmentationIdArr = vtkSmartPointer<vtkIntArray>::New();
	SegmentationIdArr->DeepCopy(contourTree->GetOutput(2)->GetPointData()->GetArray("SegmentationId"));
	contourTree->GetOutput(2)->ReleaseData();
	dataset->GetPointData()->AddArray(SegmentationIdArr);
	treeArcs->DeepCopy(contourTree->GetOutput(1));
	treeNodes->DeepCopy(contourTree->GetOutput(0));
}

void getRegSegmentation(vtkDataSet* dataset, std::unordered_set<int>& seedRegionIds, std::string& scalarName)
{
	/*
	// 1. Taking scalar's backup
	vtkNew<vtkFloatArray> scalarArray;
	scalarArray->SetName(scalarName.c_str());
	scalarArray->DeepCopy(dataset->GetPointData()->GetArray(scalarName.c_str()));

	// 2. computing the persistence diagram
	vtkSmartPointer<ttkPersistenceDiagram> diagram = vtkSmartPointer<ttkPersistenceDiagram>::New();
	diagram->SetInputData(dataset);
	diagram->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, scalarName.c_str());
	diagram->Update();

	// 3. selecting only minima-saddle pairs
	vtkSmartPointer<vtkThreshold> typePairs = vtkSmartPointer<vtkThreshold>::New();
	typePairs->SetInputConnection(diagram->GetOutputPort());
	typePairs->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "PairType");
	typePairs->SetLowerThreshold(0);
	typePairs->SetUpperThreshold(0);
	typePairs->Update();

	// 4. selecting the critical point pairs
	vtkSmartPointer<vtkThreshold> criticalPairs = vtkSmartPointer<vtkThreshold>::New();
	criticalPairs->SetInputConnection(typePairs->GetOutputPort());
	criticalPairs->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "PairIdentifier");
	criticalPairs->SetLowerThreshold(-0.1);
	criticalPairs->SetUpperThreshold(typePairs->GetOutput()->GetCellData()->GetArray("PairIdentifier")->GetRange()[1]);
	criticalPairs->Update();

	float persLowerThresh;
	//Persistence Range
	double persistence_range[2];
	criticalPairs->GetOutput()->GetCellData()->GetArray("Persistence")->GetRange(persistence_range);
	// persLowerThresh = persistence_range[1] / 1000;
	persLowerThresh = persistence_range[0];
	// cout << "Persistence Range: " << persLowerThresh << " " << persistence_range[1] << endl;
	// logFile << "Persistence Range: " << persLowerThresh << " " << persistence_range[1] << endl;

	// 5. selecting the most persistent pairs
	vtkSmartPointer<vtkThreshold> persistentPairs = vtkSmartPointer<vtkThreshold>::New();
	persistentPairs->SetInputConnection(criticalPairs->GetOutputPort());
	persistentPairs->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "Persistence");
	persistentPairs->SetLowerThreshold(persLowerThresh);
	persistentPairs->SetUpperThreshold(persistence_range[1]);
	persistentPairs->Update();

	performTopologicalSimplification(dataset, persistentPairs->GetOutput(), scalarName);
	*/

	vtkSmartPointer<vtkUnstructuredGrid> treeArcs = vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtkSmartPointer<vtkUnstructuredGrid> treeNodes = vtkSmartPointer<vtkUnstructuredGrid>::New();
	getContourTree(dataset, treeArcs, treeNodes, scalarName);

	vtkDataArray* segIdArray = treeArcs->GetCellData()->GetArray("SegmentationId");
	vtkDataArray* downNodeIdArray = treeArcs->GetCellData()->GetArray("downNodeId");
	vtkDataArray* upNodeIdArray = treeArcs->GetCellData()->GetArray("upNodeId");
	// vtkDataArray* regionSizeArray = treeArcs->GetCellData()->GetArray("RegionSize");
	vtkDataArray* criticalTypeArray = treeNodes->GetPointData()->GetArray("CriticalType");

	/*
	// Count the number of points in each segment
	std::unordered_map<int, int> regionPtCounts;
	for (int i = 0; i < regionSegmentationArray->GetNumberOfTuples(); ++i) {
		int regId = regionSegmentationArray->GetTuple1(i);
		regionPtCounts[regId]++;
	}

	// Save the ids of the regions which have less than 1 point
	std::unordered_set<int> smallRegions;
	for (std::pair<int, int> pair : regionPtCounts) {
		if (pair.second <= 1) {
			smallRegions.insert(pair.first);
		}
	}
	*/
	vtkNew<vtkIdList> arcNodeIds;
	for (int cellId = 0; cellId < treeArcs->GetNumberOfCells(); cellId++)
	{
		/*
		int regionSize = regionSizeArray->GetTuple1(cellId);
		if (regionSize < 1) {
			continue;
		}
		*/
		int downNodeid = downNodeIdArray->GetTuple1(cellId);
		int upNodeid = upNodeIdArray->GetTuple1(cellId);
		if (criticalTypeArray->GetTuple1(downNodeid) == 0 && criticalTypeArray->GetTuple1(upNodeid) == 1)
		{
			int segId = segIdArray->GetTuple1(cellId);
			seedRegionIds.insert(segId);
			/*
			// If the segment is not in the small regions
			if (smallRegions.count(segId) == 0) {
				seedRegionIds.insert(segId);
			}
			*/
		}
	}

	// dataset->GetPointData()->RemoveArray(scalarName.c_str());
	// dataset->GetPointData()->AddArray(scalarArray);
	// dataset->Modified();

	assignSegmentationCellIds(dataset);
}

void layering(vtkDataSet* dataset, std::unordered_set<int>& seedRegionIds)
{
	// getTetraMappings(dataset);
	// Create a cell array for the color ids of cells
	vtkDataArray* segmentationCellIdArr = dataset->GetCellData()->GetArray("SegmentationCellId");

	vtkNew<vtkIdList> cellPoints, pointCells;
	int pointColorId;
	segmentationCellIdArr->Modified();

	double range[2];
	segmentationCellIdArr->GetRange(range);
	// cout << "Range " << range[0] << " " << range[1] << endl;
	// logFile << "Range " << range[0] << " " << range[1] << endl;

	int iteration = 1, totalCells = 0;

	int startId, colorId, cellId, ptId;
	while (true)
	{
		std::vector<std::pair<int, int>> changedColorIds;

		// Here we first get the cellIds of those cells which are the immediate neighbors of the cells
		// already segmented.
		// std::vector<std::pair<int, int>> cellPtsSegIds;
		std::unordered_map<int, int> cellPtsSegIds;
		for (int i = 0; i < dataset->GetNumberOfCells(); i++)
		{
			// if (segmentationCellIdArr->GetTuple1(i) != queryRegId)
			if (seedRegionIds.count(segmentationCellIdArr->GetTuple1(i)) >= 1)
			{
				continue;
			}
			cellPtsSegIds.clear();
			// getImmediateNeighbors(dataset, i, pointCells);
			// getImmediateNeighborsEdge(dataset, i, pointCells);
			getImmediateNeighborsFace(dataset, i, pointCells);

			for (int j = 0; j < pointCells->GetNumberOfIds(); j++)
			{
				cellId = pointCells->GetId(j);
				colorId = segmentationCellIdArr->GetTuple1(cellId);
				if (seedRegionIds.count(colorId) >= 1)
				{
					cellPtsSegIds[colorId]++;
				}
			}

			int maxSegId, maxSegCount = -1;
			for (std::pair<int, int> pair : cellPtsSegIds)
			{
				if (pair.second > maxSegCount)
				{
					maxSegId = pair.first;
					maxSegCount = pair.second;
				}
			}

			if (maxSegCount == -1)
			{
				continue;
			}

			std::pair<int, int> pair;
			pair.first = i;
			pair.second = maxSegId;
			changedColorIds.push_back(pair);
		}

		if (changedColorIds.size() < 1)
		{
			break;
		}

		for (std::pair<int, int>& pair : changedColorIds)
		{
			segmentationCellIdArr->SetTuple1(pair.first, pair.second);
		}

		// Here we fix the problem of tetra belonging to one voxel being assigned different id.
		// This causes the regions to have pointy ends due to the nature of tetrahedral cells.
		// updateTetraIds(dataset, segmentationCellIdArr, changedColorIds);
		// updateTetraCellIds(dataset, regColorIds, segmentationCellIdArr, logFile);
		totalCells = totalCells + changedColorIds.size();
		/*
		// Count the number of cells after the update
		for (int i = 0; i < segmentationCellIdArr->GetNumberOfTuples(); i++)
		{
			int id = segmentationCellIdArr->GetTuple1(i);
			if (id != queryRegId)
			{
				totalCells += 1;
			}
		}
		cout << "Iteration: " << iteration << " totalCells: " << totalCells << " CellsRemaining: "
			<< dataset->GetNumberOfCells() - totalCells << " #queryCells " << changedColorIds.size() << endl;
		*/
		iteration++;
	}
	// updateTetraCellIds(dataset, segmentationCellIdArr);
	segmentationCellIdArr->Modified();
	segmentationCellIdArr->GetRange(range);
	// cout << "Range " << range[0] << " " << range[1] << endl;
	// logFile << "Range " << range[0] << " " << range[1] << endl;
}

void triangulate(vtkUnstructuredGrid* dataset) {
	vtkSmartPointer<vtkDataSetTriangleFilter> tetrahedralize =
		vtkSmartPointer<vtkDataSetTriangleFilter>::New();
	tetrahedralize->SetInputData(dataset);
	tetrahedralize->Update();
	dataset->DeepCopy(tetrahedralize->GetOutput());
}

void colorRegions(vtkUnstructuredGrid* dataset, vtkDataSet* DATASET, std::string scalarName, std::string segArrayName,
	bool isManifold = false)
{
	if (isManifold == false) {
		triangulate(dataset);
		getManifoldMultiRegion(dataset);
		dataset->Delete();
	}

	// double scalar_range[2];
	// dataset->GetPointData()->GetArray(scalarName.c_str())->GetRange(scalar_range);
	// cout << scalar_range[0] << " " << scalar_range[1] << endl;

	int MAX_ID = 0;
	// Main loop to traverse through each region
	int count = 0;
	for (auto& pair : REGIONS)
	{
		count++;
		vtkSmartPointer<vtkUnstructuredGrid> regionDataset = pair.second;

		vtkSmartPointer<vtkIntArray> regColorIds = vtkSmartPointer<vtkIntArray>::New();
		regColorIds->SetNumberOfComponents(1);
		regColorIds->SetNumberOfTuples(regionDataset->GetNumberOfCells());
		regColorIds->SetName(segArrayName.c_str());

		std::unordered_set<int> seedRegionIds;
		getRegSegmentation(regionDataset, seedRegionIds, scalarName);
		layering(regionDataset, seedRegionIds);
		// Assign the color ids
		vtkDataArray* segmentationCellIdArr = regionDataset->GetCellData()->GetArray("SegmentationCellId");
		// vtkDataArray* globalCellIdsArr = regionDataset->GetCellData()->GetArray("GlobalCellIds");
		for (int cellId = 0; cellId < regionDataset->GetNumberOfCells(); cellId++)
		{
			// int globalCellId = globalCellIdsArr->GetTuple1(cellId);
			int segId = segmentationCellIdArr->GetTuple1(cellId);
			regColorIds->SetTuple1(cellId, segId + MAX_ID);
		}
		regColorIds->Modified();
		// cout << regColorIds->GetRange()[0] << " " << regColorIds->GetRange()[1] << endl;
		MAX_ID = regColorIds->GetRange()[1] + 1;
		cout << count << " " << REGIONS.size() << endl;
		regionDataset->GetCellData()->RemoveArray("SegmentationCellId");
		regionDataset->GetCellData()->AddArray(regColorIds);
	}

	vtkSmartPointer<vtkAppendDataSets> append = vtkSmartPointer<vtkAppendDataSets>::New();
	for (auto& pair : REGIONS)
	{
		vtkSmartPointer<vtkUnstructuredGrid> temp = pair.second;
		append->AddInputData(temp);
	}
	append->Update();

	vtkDataArray* colorIdsArray = append->GetOutput()->GetCellData()->GetArray(segArrayName.c_str());

	vtkSmartPointer<vtkIntArray> regColorIds = vtkSmartPointer<vtkIntArray>::New();
	regColorIds->SetNumberOfComponents(1);
	regColorIds->SetNumberOfTuples(DATASET->GetNumberOfCells());
	regColorIds->SetName(segArrayName.c_str());

	vtkPointSet* PS = vtkPointSet::SafeDownCast(append->GetOutput());
	PS->BuildCellLocator();
	PS->EditableOff();

	vtkSmartPointer<vtkAbstractCellLocator> CellLocator = PS->GetCellLocator();

	vtkSmartPointer<vtkCellCenters> cellCenters = vtkSmartPointer<vtkCellCenters>::New();
	cellCenters->SetInputData(DATASET);
	cellCenters->Update();

	vtkDataSet* centers = cellCenters->GetOutput();

	for (int i = 0; i < DATASET->GetNumberOfCells(); i++)
	{
		double pt[3];
		centers->GetPoint(i, pt);
		double closestPoint[3];
		vtkIdType cellId;
		int subId;
		double dist2;
		CellLocator->FindClosestPoint(pt, closestPoint, cellId, subId, dist2);
		int colorId = colorIdsArray->GetTuple1(cellId);
		regColorIds->SetTuple1(i, colorId);
	}
	DATASET->GetCellData()->AddArray(regColorIds);
	DATASET->GetCellData()->SetActiveScalars(regColorIds->GetName());
}

int main(int argc, char* argv[])
{
	vtkObject::GlobalWarningDisplayOff();
	// vtkSMPTools::SetBackend("STDThread");
	// vtkSMPTools::Initialize(8);

	int posOrNeg = 0;
	// Requires dataset_Regions, datasetFull.csv, dataset_SplitRegions.vtk and
	// dataset_SplitRegions.json file

	if (argc < 2)
	{
		std::cerr << "Please specify the input filename." << endl;
		exit(1);
	}

	vtkNew<vtkNamedColors> colors;

	// std::string datafile = "./initial_value_problem/fortRegionGrowing_float_bin.vtk";

	std::string datafile = argv[1];

	vtkSmartPointer<vtkDataSetReader> dataReader = vtkSmartPointer<vtkDataSetReader>::New();
	dataReader->SetFileName(datafile.c_str());
	dataReader->Update();
	vtkDataSet* DATASET = dataReader->GetOutput();
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	vtkNew<vtkUnstructuredGrid> dataset;
	dataset->DeepCopy(dataReader->GetOutput());

	std::string segArrayName = "RegionIds";
	std::string scalarName = "vorticity_mag";
	colorRegions(dataset, DATASET, scalarName, segArrayName);

	segArrayName = "SegmentIds";
	scalarName = "vorticity_mag";
	colorRegions(dataset, DATASET, scalarName, segArrayName, true);
	// dataset->Delete();

	vtkDataArray* scalarThresholds = DATASET->GetFieldData()->GetArray("ScalarThreshold");
	double isoValue = scalarThresholds->GetTuple(0)[1];

	vtkSmartPointer<vtkContourFilter> contourFilter = vtkSmartPointer<vtkContourFilter>::New();
	contourFilter->SetInputData(DATASET);
	contourFilter->SetInputArrayToProcess(0, 0, 0,
		vtkDataObject::FIELD_ASSOCIATION_POINTS, "lambda2");
	contourFilter->SetValue(0, isoValue);
	contourFilter->Update();
	vtkDataSet* isoSurfaces = contourFilter->GetOutput();

	while (isoSurfaces->GetPointData()->GetNumberOfArrays() != 0) {
		isoSurfaces->GetPointData()->RemoveArray(0);
	}

	while (isoSurfaces->GetCellData()->GetNumberOfArrays() != 0) {
		isoSurfaces->GetCellData()->RemoveArray(0);
	}

	vtkDataArray* colorIdsArray = DATASET->GetCellData()->GetArray(segArrayName.c_str());

	vtkSmartPointer<vtkIntArray> regColorIds = vtkSmartPointer<vtkIntArray>::New();
	regColorIds->SetNumberOfComponents(1);
	regColorIds->SetNumberOfTuples(isoSurfaces->GetNumberOfCells());
	regColorIds->SetName("SurfaceRegionIds");

	vtkPointSet* PS = vtkPointSet::SafeDownCast(DATASET);
	PS->BuildCellLocator();
	PS->EditableOff();

	vtkSmartPointer<vtkAbstractCellLocator> CellLocator = PS->GetCellLocator();

	vtkSmartPointer<vtkCellCenters> surfaceCellCenters = vtkSmartPointer<vtkCellCenters>::New();
	surfaceCellCenters->SetInputData(isoSurfaces);
	surfaceCellCenters->Update();

	vtkDataSet* centers = surfaceCellCenters->GetOutput();

	for (int i = 0; i < isoSurfaces->GetNumberOfCells(); i++)
	{
		double pt[3];
		centers->GetPoint(i, pt);
		vtkSmartPointer<vtkGenericCell> localCell = vtkSmartPointer<vtkGenericCell>::New();
		double pcoords[3];
		double weights[8];
		// double closestPoint[3];
		// vtkIdType cellId;
		// int subId;
		// double dist2;
		vtkIdType cellId = CellLocator->FindCell(pt, CellLocator->GetTolerance(), localCell, pcoords, weights);
		// CellLocator->FindClosestPoint(pt, closestPoint, cellId, subId, dist2);
		if (cellId == -1) {
			cout << i << " ";
			continue;
		}
		int colorId = colorIdsArray->GetTuple1(cellId);
		regColorIds->SetTuple1(i, colorId);
	}

	isoSurfaces->GetCellData()->AddArray(regColorIds);
	isoSurfaces->GetCellData()->SetActiveScalars(regColorIds->GetName());

	isoSurfaces->GetCellData()->RemoveArray(segArrayName.c_str());

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
	logFile << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

	std::string temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_Split.vtk";

	vtkSmartPointer<vtkDataSetWriter> writer = vtkSmartPointer<vtkDataSetWriter>::New();
	writer->SetInputData(DATASET);
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_Split_Isosurfaces.vtk";

	writer->SetInputData(isoSurfaces);
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	return EXIT_SUCCESS;
}