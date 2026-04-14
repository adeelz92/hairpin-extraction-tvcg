// In this version, the extracted regions are of type tetrahedral grids.

// Region_Growing.exe "Path\To\FullDataVTKfile.vtk"
#include <vtkNew.h>
#include <vtkContourFilter.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkGradientFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetReader.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkThreshold.h>
#include <vtkConnectivityFilter.h>
#include <vtkAppendFilter.h>
#include <vtkDataSetWriter.h>
#include <vtkStructuredGrid.h>
#include <vtkStaticCellLocator.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

struct MyRange
{
	double min;
	double max;


	MyRange()
	{
		min = 0;
		max = 0;
	}

	MyRange(double _min, double _max)
	{
		min = _min;
		max = _max;
	}
};

void getCellsList(vtkDataSet* dataset, std::vector<std::array<double, 3>> positions, vtkStaticCellLocator* CellLocator,
	vtkIdList* ptsList, vtkDataArray* lambda2, double scalar_threshold)
{
	size_t size = positions.size();
	for (int pointId = 0; pointId < size; pointId++)
	{
		double *position = positions.at(pointId).data();
		// Get cell id
		vtkIdType Id = CellLocator->FindCell(position);
		if (Id >= 0)
		{
			// Get cell
			vtkNew<vtkGenericCell> cell;
			dataset->GetCell(Id, cell);

			// Get Cell Points
			vtkIdList* cellPtIds = cell->GetPointIds();

			for (int k = 0; k < cellPtIds->GetNumberOfIds(); k++)
			{
				vtkIdType id = cellPtIds->GetId(k);
				double value = *lambda2->GetTuple(id);

				if (value <= scalar_threshold)
				{
					ptsList->InsertUniqueId(Id);
					break;
				}
			}
		}
	}
}

void getCellsList(int window_size, int xDims, int yDims, int zDims,
	vtkDataSet* input, double scalar_threshold, std::unordered_set<vtkIdType>& cellsList, std::string& scalarName)
{
	vtkDataArray* scalarArray = input->GetPointData()->GetArray(scalarName.c_str());

	for (int z_idx = 0; z_idx < (zDims - (window_size - 1)); z_idx += window_size)
	{
		vtkIdType* z_pts = new vtkIdType[window_size];
		for (int ii = 0; ii < window_size; ii++)
		{
			z_pts[ii] = z_idx + ii;
		}

		for (int y_idx = 0; y_idx < (yDims - (window_size - 1)); y_idx += window_size)
		{
			vtkIdType* y_pts = new vtkIdType[window_size];
			for (int ii = 0; ii < window_size; ii++)
			{
				y_pts[ii] = y_idx + ii;
			}

			for (int x_idx = 0; x_idx < (xDims - (window_size - 1)); x_idx += window_size)
			{
				vtkIdType* x_pts = new vtkIdType[window_size];
				for (int ii = 0; ii < window_size; ii++)
				{
					x_pts[ii] = x_idx + ii;
				}

				double minValue = scalar_threshold;
				vtkIdType minId = -1;
				vtkIdType minCellId = -1;

				for (int k = 0; k < window_size; k++)
				{
					for (int j = 0; j < window_size; j++)
					{
						for (int i = 0; i < window_size; i++)
						{
							vtkIdType pointId = x_pts[i] + (xDims * y_pts[j]) + (xDims * yDims * z_pts[k]);
							double value = scalarArray->GetTuple1(pointId);
							if (value <= minValue)
							{
								vtkSmartPointer<vtkIdList> ptCellIds = vtkSmartPointer<vtkIdList>::New();
								input->GetPointCells(pointId, ptCellIds);

								minValue = value;
								minId = pointId;
								minCellId = ptCellIds->GetId(0);
							}
						}
					}
				}
				delete[] x_pts;

				if (minCellId != -1)
				{
					cellsList.insert(minCellId);
				}
			}
			delete[] y_pts;
		}
		delete[] z_pts;
	}
}

void addPtIdxLists(int xDims, int yDims, int zDims, vtkDataSet* input)
{
	vtkSmartPointer<vtkUnsignedIntArray> PtIdsArray =
		vtkSmartPointer<vtkUnsignedIntArray>::New();
	PtIdsArray->SetNumberOfComponents(1);
	PtIdsArray->SetNumberOfTuples(input->GetNumberOfPoints());
	PtIdsArray->SetName("PointIds");
	for (unsigned int i = 0; i < input->GetNumberOfPoints(); ++i) {
		PtIdsArray->SetTuple1(i, i);
	}
	input->GetPointData()->AddArray(PtIdsArray);

	/*
	vtkSmartPointer<vtkIntArray> xIdxArray = 
		vtkSmartPointer<vtkIntArray>::New();
	xIdxArray->SetNumberOfComponents(1);
	xIdxArray->SetNumberOfTuples(input->GetNumberOfPoints());
	xIdxArray->SetName("X");

	vtkSmartPointer<vtkIntArray> yIdxArray =
		vtkSmartPointer<vtkIntArray>::New();
	yIdxArray->SetNumberOfComponents(1);
	yIdxArray->SetNumberOfTuples(input->GetNumberOfPoints());
	yIdxArray->SetName("Y");

	vtkSmartPointer<vtkIntArray> zIdxArray =
		vtkSmartPointer<vtkIntArray>::New();
	zIdxArray->SetNumberOfComponents(1);
	zIdxArray->SetNumberOfTuples(input->GetNumberOfPoints());
	zIdxArray->SetName("Z");

	for (int z_idx = 0; z_idx < zDims; ++z_idx)
	{
		for (int y_idx = 0; y_idx < yDims; ++y_idx)
		{
			for (int x_idx = 0; x_idx < xDims; ++x_idx)
			{
				int pointId = x_idx + (xDims * y_idx) + (xDims * yDims * z_idx);
				xIdxArray->SetTuple1(pointId, x_idx);
				yIdxArray->SetTuple1(pointId, y_idx);
				zIdxArray->SetTuple1(pointId, z_idx);
			}
		}
	}

	input->GetPointData()->AddArray(xIdxArray);
	input->GetPointData()->AddArray(yIdxArray);
	input->GetPointData()->AddArray(zIdxArray);
	*/
}

void calculateVelUf(int xDims, int yDims, int zDims, vtkDataSet* input,
	vtkFloatArray* vel_uf_array)
{
	vtkDataArray* velocityArray = input->GetPointData()->GetArray("velocity");
	double vel_value[3];
	double avg_u;
	for (int k = 0; k < zDims; k++)
	{
		avg_u = 0.0;
		// Compute the average oyf value for 1 plane
		for (int j = 0; j < yDims; j++)
		{
			for (int i = 0; i < xDims; i++)
			{
				vtkIdType pointId = i + (xDims * j) + (xDims * yDims * k);
				velocityArray->GetTuple(pointId, vel_value);
				avg_u += vel_value[0]; // Compute Average oy
			}
		}
		avg_u = avg_u / (xDims * yDims);

		// Assign a value to oyf_array oyf = oy - avg_oy
		for (int jj = 0; jj < yDims; jj++)
		{
			for (int ii = 0; ii < xDims; ii++)
			{
				vtkIdType pointId = ii + (xDims * jj) + (xDims * yDims * k);
				velocityArray->GetTuple(pointId, vel_value);
				double u = vel_value[0];
				double v = vel_value[1];
				double w = vel_value[2];
				double uF = u - avg_u;
				vel_uf_array->SetTuple3(pointId, uF, v, w);
			}
		}
	}
}

void addVorticityMagnitude(vtkDataSet* dataset)
{
	if (!dataset->GetPointData()->HasArray("vorticity")) {
		vtkSmartPointer<vtkGradientFilter> gradient = vtkSmartPointer<vtkGradientFilter>::New();
		gradient->SetInputData(dataset);
		gradient->ComputeGradientOff();
		gradient->ComputeDivergenceOff();
		gradient->ComputeQCriterionOff();
		gradient->ComputeVorticityOn();
		gradient->SetVorticityArrayName("vorticity");
		gradient->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "velocity");
		gradient->Update();
		vtkSmartPointer<vtkFloatArray> vorticityArr = vtkSmartPointer<vtkFloatArray>::New();
		vorticityArr->DeepCopy(gradient->GetOutput()->GetPointData()->GetArray("vorticity"));
		dataset->GetPointData()->AddArray(vorticityArr);

		vtkSmartPointer<vtkFloatArray> vorticity_mag = vtkSmartPointer<vtkFloatArray>::New();
		vorticity_mag->SetNumberOfComponents(1);
		vorticity_mag->SetNumberOfTuples(dataset->GetNumberOfPoints());
		vorticity_mag->SetName("vorticity_mag");
		double vorticity[3];
		for (vtkIdType i = 0; i < dataset->GetNumberOfPoints(); ++i) {
			vorticityArr->GetTuple(i, vorticity);
			vorticity_mag->SetTuple1(i, vtkMath::Norm(vorticity));
		}
		dataset->GetPointData()->AddArray(vorticity_mag);
		dataset->GetPointData()->RemoveArray("vorticity");
	}
}

void addVorticityArray(vtkDataSet* dataset)
{
	vtkSmartPointer<vtkGradientFilter> gradient = vtkSmartPointer<vtkGradientFilter>::New();
	gradient->SetInputData(dataset);
	gradient->ComputeGradientOff();
	gradient->ComputeDivergenceOff();
	gradient->ComputeQCriterionOff();
	gradient->ComputeVorticityOn();
	gradient->SetVorticityArrayName("vorticity");
	gradient->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "velocity");
	gradient->Update();
	vtkSmartPointer<vtkFloatArray> vorticityArr = vtkSmartPointer<vtkFloatArray>::New();
	vorticityArr->DeepCopy(gradient->GetOutput()->GetPointData()->GetArray("vorticity"));
	dataset->GetPointData()->AddArray(vorticityArr);

	/*
	vtkSmartPointer<vtkFloatArray> vorticity_mag = vtkSmartPointer<vtkFloatArray>::New();
	vorticity_mag->SetNumberOfComponents(1);
	vorticity_mag->SetNumberOfTuples(dataset->GetNumberOfPoints());
	vorticity_mag->SetName("vorticity_mag");
	double vorticity[3];
	for (vtkIdType i = 0; i < dataset->GetNumberOfPoints(); ++i) {
		vorticityArr->GetTuple(i, vorticity);
		vorticity_mag->SetTuple1(i, vtkMath::Norm(vorticity));
	}
	dataset->GetPointData()->AddArray(vorticity_mag);
	*/
}

void addVorticityUfArray(vtkDataSet* dataset)
{
	// Compute & Generate vorticity with velocity-uf
	vtkSmartPointer<vtkGradientFilter> gradient = vtkSmartPointer<vtkGradientFilter>::New();
	gradient->SetInputData(dataset);
	gradient->ComputeGradientOff();
	gradient->ComputeDivergenceOff();
	gradient->ComputeQCriterionOff();
	gradient->ComputeVorticityOn();
	gradient->SetVorticityArrayName("vorticity-uf");
	gradient->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "velocity-uf");
	gradient->Update();
	vtkDataArray* vorticity_uf = gradient->GetOutput()->GetPointData()->GetArray("vorticity-uf");
	dataset->GetPointData()->AddArray(vorticity_uf);
	gradient->GetOutput()->ReleaseData();
}

void getConnectivity(vtkDataSet* finalGrid, vtkUnstructuredGrid* grid, std::unordered_set<vtkIdType>& cellsList, double scalar_range[2]) {
	vtkSmartPointer<vtkConnectivityFilter> connectivityFilter = vtkSmartPointer<vtkConnectivityFilter>::New();
	connectivityFilter->SetInputData(finalGrid);
	connectivityFilter->ScalarConnectivityOn();
	connectivityFilter->ColorRegionsOff();
	connectivityFilter->SetScalarRange(scalar_range);
	connectivityFilter->SetExtractionModeToCellSeededRegions();
	connectivityFilter->InitializeSeedList();

	for (vtkIdType cellIdx : cellsList)
	{
		connectivityFilter->AddSeed(cellIdx);
	}
	connectivityFilter->Update();
	grid->DeepCopy(connectivityFilter->GetOutput());
}

int main(int argc, char* argv[])
{
	// vtkSMPTools::SetBackend("STDThread");
	// vtkSMPTools::Initialize(2);

	if (argc < 2)
	{
		std::cerr << "Please specify input file." << endl;
		exit(1);
	}
	// std::string datafile = "./fort_data/fort180_full_float_bin.vtk";

	std::string scalarname = "lambda2";
	int posOrNeg = 0;
	std::string datafile = argv[1];
	vtkNew<vtkDataSetReader> dataReader;
	dataReader->SetFileName(datafile.c_str());
	dataReader->Update();

	vtkSmartPointer<vtkStructuredGrid> dataset = vtkStructuredGrid::SafeDownCast(dataReader->GetOutput());

	dataset->GetPointData()->SetActiveScalars(scalarname.c_str());

	addVorticityMagnitude(dataset);
	// addVorticityArray(dataset);

	double scalar_range[2];
	dataset->GetScalarRange(scalar_range);

	scalar_range[1] = std::stof(argv[2]);

	cout << scalar_range[0] << " " << scalar_range[1] << endl;
	
	double dataBounds[6];
	dataset->GetBounds(dataBounds);
	double xMin = dataBounds[0], xMax = dataBounds[1];
	double yMin = dataBounds[2], yMax = dataBounds[3];
	double zMin = dataBounds[4], zMax = dataBounds[5];

	cout << xMin << " " << xMax << " " << yMin << " " << yMax << " " << zMin << " " << zMax << endl;
	int dims[3];
	dataset->GetDimensions(dims);
	int xDim = dims[0]; // getDims(dataset, dataBounds, 'x');
	int yDim = dims[1];  // getDims(dataset, dataBounds, 'y');
	int zDim = dims[2]; // getDims(dataset, dataBounds, 'z');

	cout << "xDim: " << xDim << endl;
	cout << "yDim: " << yDim << endl;
	cout << "zDim: " << zDim << endl;

	// addPtIdxLists(xDim, yDim, zDim, dataset);


	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	vtkSmartPointer<vtkDataSet> finalGrid = vtkDataSet::SafeDownCast(dataset);
	std::unordered_set<vtkIdType> cellsList;
	getCellsList(5, xDim, yDim, zDim, finalGrid, scalar_range[1], cellsList, scalarname);
	
	vtkNew<vtkUnstructuredGrid> Regions;
	getConnectivity(finalGrid, Regions, cellsList, scalar_range);

	addVorticityArray(Regions);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

	vtkSmartPointer<vtkIntArray> dimsArray = vtkSmartPointer<vtkIntArray>::New();
	dimsArray->SetName("StructuredGridDimensions");
	dimsArray->SetNumberOfComponents(3);
	dimsArray->SetNumberOfTuples(1);
	dimsArray->InsertTypedTuple(0, dims);
	Regions->GetFieldData()->AddArray(dimsArray);

	vtkSmartPointer<vtkDoubleArray> ScalarThreshold = vtkSmartPointer<vtkDoubleArray>::New();
	ScalarThreshold->SetName("ScalarThreshold");
	ScalarThreshold->SetNumberOfComponents(2);
	ScalarThreshold->SetNumberOfTuples(1);
	ScalarThreshold->InsertTypedTuple(0, scalar_range);
	Regions->GetFieldData()->AddArray(ScalarThreshold);

	std::string temp = argv[1];
	temp = temp.erase(temp.length() - 4);
	temp = temp + "_Regions.vtk";
	vtkNew<vtkDataSetWriter> writer;
	writer->SetInputData(Regions);
	writer->SetFileName(temp.c_str());
	writer->SetFileTypeToBinary();
	writer->Write();

	return EXIT_SUCCESS;
}
