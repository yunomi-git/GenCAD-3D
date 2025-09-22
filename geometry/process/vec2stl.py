import os
from pathlib import Path
import h5py
from OCC.Extend.DataExchange import write_stl_file, write_step_file
import trimesh
import geometry.trimesh_util as trimesh_util
import numpy as np
from tqdm import tqdm

import paths

from cadlib.visualize import vec2CADsolid
from geometry.geometry_data import GeometryLoader
from utils.multi_proc_queue import MultiProcQueueProcessing, ERROR_STATUS
from paths import DirectoryPathManager
from .process_geometry import load_stl_as_cloud


def vec_to_stl(input_directory, output_stl_directory, output_pointcloud_directory, normalize=False):
    directory_manager = DirectoryPathManager(input_directory, base_unit_is_file=True)
    # filter for h5 files
    cad_vec_paths = directory_manager.file_paths
    cad_vec_paths = [file_path for file_path in cad_vec_paths if file_path.extension == ".h5"]

    def task(vec_path, output_stl_directory, output_pointcloud_directory):
        try:
            with h5py.File(vec_path.as_absolute(), 'r') as fp:
                # Grab CAD
                out_vec = fp["vec"][:].astype(float)
                out_shape = vec2CADsolid(out_vec)

                # Grab mesh
                Path(output_stl_directory + vec_path.subfolder_path).mkdir(parents=True, exist_ok=True)
                out_file = f"{output_stl_directory}/{vec_path.as_relative(extension=False)}.stl"
                write_stl_file(out_shape, out_file,
                               mode="binary",
                               linear_deflection=0.001,
                               angular_deflection=0.1)

                # Now grab pointcloud by sampling from mesh
                cloud = load_stl_as_cloud(out_file)
                if cloud is None:
                    return ERROR_STATUS

                Path(output_pointcloud_directory + vec_path.subfolder_path).mkdir(parents=True, exist_ok=True)
                out_file = f"{output_pointcloud_directory}/{vec_path.as_relative(extension=False)}"
                np.save(out_file, cloud)

        except Exception as e:
            # print('cannot create stl:', e)
            return ERROR_STATUS

    multiproc = MultiProcQueueProcessing(args_global=(output_stl_directory, output_pointcloud_directory), task=task, num_workers=16)
    multiproc.process(process_list=cad_vec_paths)

def vec_to_step(geometry_loader: GeometryLoader):
    output_step_directory = geometry_loader.step_dir

    for data_id in tqdm(geometry_loader.get_all_valid_data(cad=True)):
        try:
            out_file = f"{output_step_directory}{data_id}.step"
            if os.path.isfile(out_file):
                continue

            cad_vec = geometry_loader.load_cad(data_id, pad=False, tensor=False, as_single_vec=True)
            out_shape = vec2CADsolid(cad_vec)

            # Grab mesh
            folder = data_id[:4]
            Path(output_step_directory + folder).mkdir(parents=True, exist_ok=True)

            write_step_file(out_shape, out_file)
        except Exception as e:
            print('cannot create step:', e, data_id)
            continue

import argparse
if __name__ == "__main__":
    # Input: data_path/name/cad_vec/
    # output: data_path/name/stls
    # output: data_path/name/clouds
    parser = argparse.ArgumentParser(description="Convert cad vecs to STL and clouds.")
    parser.add_argument("-name", "--dataset_name", type=str, required=True, help="name of cad dataset")
    args = parser.parse_args()
    # normalize = False

    data_root = paths.DATA_PATH + args.dataset_name + "/"
    input_dir = data_root + "cad_vec/"

    last_folder_index = input_dir[:-1].rfind("/")
    output_stl_dir = input_dir[:last_folder_index] + "/stls/"
    # output_step_dir = input_dir[:last_folder_index] + "/steps/"

    output_pc_dir = input_dir[:last_folder_index] + "/clouds/"

    Path(output_stl_dir).mkdir(parents=True, exist_ok=True)
    Path(output_pc_dir).mkdir(parents=True, exist_ok=True)
    # Path(output_step_dir).mkdir(parents=True, exist_ok=True)

    print("Outputting to:")
    print(output_stl_dir)
    print(output_pc_dir)
    # print(output_step_dir)

    vec_to_stl(input_dir, output_stl_dir, output_pc_dir)

    # geometry_loader = GeometryLoader(data_root=data_root, phase=None, splits_file="filtered_data_deepcad")
    # vec_to_step(geometry_loader)


    print("#" * 50)
    print("         PC creation completed        ")
    print("#" * 50)
