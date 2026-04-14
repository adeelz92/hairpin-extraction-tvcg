import vtk


def assign_point_ids_by_nearest(main_polydata, candidates_polydata,
                                point_id_array_name="PointIds"):
    """
    Assigns PointIds from main_polydata to candidates_polydata
    using nearest-point interpolation.
    """

    # --- Get PointIds array from main dataset ---
    main_ids = main_polydata.GetPointData().GetArray(point_id_array_name)
    if main_ids is None:
        raise RuntimeError(f"Array '{point_id_array_name}' not found in main dataset")

    # --- Build point locator on main dataset ---
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(main_polydata)
    locator.BuildLocator()

    # --- Create output vtkIdTypeArray ---
    out_ids = vtk.vtkIdTypeArray()
    out_ids.SetName(point_id_array_name)
    out_ids.SetNumberOfComponents(1)
    out_ids.SetNumberOfTuples(candidates_polydata.GetNumberOfPoints())

    # --- Assign nearest PointIds ---
    for i in range(candidates_polydata.GetNumberOfPoints()):
        p = candidates_polydata.GetPoint(i)

        nearest_id = locator.FindClosestPoint(p)
        original_id = main_ids.GetValue(nearest_id)

        out_ids.SetValue(i, original_id)

    # --- Attach array to candidates dataset ---
    candidates_polydata.GetPointData().AddArray(out_ids)
    # candidates_polydata.GetPointData().SetActiveScalars(point_id_array_name)

    return candidates_polydata

# Read main.vtk
reader_main = vtk.vtkDataSetReader()
reader_main.SetFileName(r"F:\Adeel\studies\UH\Research\2024\vortices_statistics\datasets\channel\channel.vtk")
reader_main.Update()
main_pd = reader_main.GetOutput()

# Read main_candidates.vtk
reader_candidates = vtk.vtkDataSetReader()
reader_candidates.SetFileName(r"F:\Adeel\studies\UH\Research\2026\hairpin-flow\datasets\channel_test\criteria_analysis\channel_test_Regions_Split_hairpinRegion.vtk")
reader_candidates.Update()
candidates_pd = reader_candidates.GetOutput()

# Assign PointIds
candidates_pd = assign_point_ids_by_nearest(
    main_pd,
    candidates_pd,
    point_id_array_name="PointIds"
)

# Write output
writer = vtk.vtkDataSetWriter()
writer.SetFileName(r"F:\Adeel\studies\UH\Research\2026\hairpin-flow\datasets\channel_test\criteria_analysis\channel_test_Regions_Split_hairpinRegion_with_ids.vtk")
writer.SetInputData(candidates_pd)
writer.SetFileTypeToBinary()
writer.Write()
