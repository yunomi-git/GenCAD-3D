import pymeshlab
from tqdm import tqdm

import paths
import utils.util as util
import trimesh
import geometry.trimesh_util as trimesh_util
from pathlib import Path
from paths import DirectoryPathManager
import argparse
import os
from multiprocessing import Queue, Process
from utils.multi_proc_queue import MultiProcQueueProcessing, ERROR_STATUS

time = util.Stopwatch()

def _remesh_default(file_name, length_scale):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_name)

    # Subdivision is important for good isotropic
    ms.meshing_surface_subdivision_midpoint(iterations=3)
    ms.meshing_isotropic_explicit_remeshing(iterations=7, targetlen=pymeshlab.PercentageValue(length_scale))

    return ms

def _remesh_nosubdivision(file_name, length_scale):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_name)

    # Subdivision is important for good isotropic
    ms.meshing_isotropic_explicit_remeshing(iterations=7, targetlen=pymeshlab.PercentageValue(length_scale))

    return ms

def _remesh_thin_rod(file_name, num_verts):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_name)

    # Goal: Subdivision multiple times to get vertices on the long faces
    # Use decimation in middle to prevent vertex count from getting too high
    ms.meshing_surface_subdivision_midpoint(iterations=3)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_verts * 2, preserveboundary=True, preservenormal=True, preservetopology=True)
    ms.meshing_surface_subdivision_midpoint(iterations=3)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_verts * 2, preserveboundary=True, preservenormal=True, preservetopology=True)
    # Now do the isotropic remeshing to get even-sized faces
    ms.meshing_surface_subdivision_midpoint(iterations=3)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_verts * 8, preserveboundary=True, preservenormal=True, preservetopology=True)
    ms.meshing_surface_subdivision_midpoint(iterations=3)
    ms.meshing_isotropic_explicit_remeshing(iterations=15, adaptive=True)

    # Finish by asserting correct size
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_verts * 2, preserveboundary=True, preservenormal=True, preservetopology=True)
    return ms


def _remesh_and_check(file_name, length_scale, min_verts, max_verts, remesh_function):
    #remesh function: f(file_name, length_scale) -> ms
    # ms = remesh(file_name, length_scale)
    ms = remesh_function(file_name, length_scale)

    num_verts = ms.current_mesh().vertex_number()
    if num_verts < min_verts:
        return -1.0
    if num_verts > max_verts:
        return 1.0
    else:
        return 0.0

def _get_remesh_comparison_function(filename, min_verts, max_verts, remesh_function):
    return lambda length_scale: _remesh_and_check(filename, length_scale, min_verts, max_verts, remesh_function)

def _binary_search(min_x, max_x, comparison_function, max_depth=3, depth=0):
    # when passed through comparison_function(x), min_x should return a smaller value than max_x
    test_value = (min_x + max_x) / 2

    if depth == max_depth:
        return test_value

    result = comparison_function(test_value)
    if result > 0:
        return _binary_search(min_x, test_value, comparison_function, max_depth, depth + 1)
    elif result < 0:
        return _binary_search(test_value, max_x, comparison_function, max_depth, depth + 1)
    else:
        return test_value


def full_remesh_and_save(input_file, save_file, min_vertices, max_vertices, remesh_function=_remesh_default):
    try:
        mesh = trimesh.load(input_file)
        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    except:
        print("error loading trimesh", input_file)
        return None

    if not mesh_aux.is_valid or not mesh.is_watertight:
        print("mesh is not valid", input_file)
        return None

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_file)
    except:
        print("error loading meshlab", input_file)
        return None

    try:
        # First try normal remeshing
        remesh_comparison_func = _get_remesh_comparison_function(filename=input_file,
                                                                 min_verts=min_vertices,
                                                                 max_verts=max_vertices,
                                                                 remesh_function=remesh_function)

        min_start_length = 4.0  # larger = fewer verts
        max_start_length = 0.01  # smaller = more verts
        length_to_use = _binary_search(min_start_length, max_start_length, comparison_function=remesh_comparison_func,
                                       max_depth=4)

        ms = remesh_function(input_file, length_to_use)

        # If it fails, then try thin rod remeshing
        # if ms.current_mesh().vertex_number() < min_vertices:
        #     print("Normal remesh failed. Doing thin remesh")
        #     ms = _remesh_thin_rod(input_file, min_vertices)

        # If it still fails, something is really wrong. Do not remesh
        if ms.current_mesh().vertex_number() < min_vertices:
            print("vertex error", input_file)
            return None

        ms.save_current_mesh(save_file, save_face_color=False)
        # do not normalize
        # mesh = trimesh.load(save_file)
        # mesh = trimesh_util.normalize_mesh(mesh, center=False, normalize_scale=False)
        # mesh.export(save_file)
        # ms.save_current_mesh(save_folder + file_name.as_relative(extension=False) + ".stl",
        #                      save_face_color=False)
        return ms.current_mesh().vertex_number()

    except Exception as e:
        print("error remeshing ", input_file, "| ", e)
        return None


def remesh_queue_task(file_name, input_dir, output_dir, min_desired_vertices, max_desired_vertices, remesh_function):
    save_name = output_dir + file_name
    input_name = input_dir + file_name

    output = full_remesh_and_save(input_file=input_name, save_file=save_name,
                         min_vertices=min_desired_vertices, max_vertices=max_desired_vertices, remesh_function=remesh_function)
    if output is None:
        return ERROR_STATUS

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Convert an input directory of STLs to remeshed versions")
    parser.add_argument("-input", "--input_folder", type=str, required=True, help="input directory of stls")
    parser.add_argument("-output", "--output_folder", type=str, required=True, help="output directory of stls")

    args = parser.parse_args()

    # desired_vertex_range = [2048, 4000]
    min_desired_vertices = 2048
    max_desired_vertices = 4000
    original_path = paths.DATA_PATH + args.input_folder #"DaVinci_CAD_Augmented_NA_copy/stls/"
    new_path = paths.DATA_PATH + args.output_folder #"temp/remeshed_stls/"
    remesh_function = _remesh_default

    Path(new_path).mkdir(exist_ok=True, parents=True)

    folders = os.listdir(original_path)
    folders.sort()
    for folder in folders[31:]:
        print("Processing folder", folder)
        root_path = original_path + folder + "/"

        directory_manager = DirectoryPathManager(root_path, base_unit_is_file=True)
        file_paths = directory_manager.file_paths

        save_folder = new_path + folder + "/"
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        multiproc = MultiProcQueueProcessing(args_global=(root_path, save_folder, min_desired_vertices, max_desired_vertices, remesh_function),
                                             task=remesh_queue_task, num_workers=22)
        input_files = directory_manager.get_files_relative()
        multiproc.process(process_list=input_files)






