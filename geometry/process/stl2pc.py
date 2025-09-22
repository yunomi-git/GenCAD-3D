import os
from pathlib import Path
import h5py
from IPython.terminal.pt_inputhooks.wx import inputhook
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


def stl2pc(input_stl_directory, output_pointcloud_directory, normalize=False):
    directory_manager = DirectoryPathManager(input_stl_directory, base_unit_is_file=True)
    # filter for h5 files
    stl_paths = directory_manager.file_paths
    stl_paths = [file_path for file_path in stl_paths if file_path.extension == ".h5"]

    def task(stl_path, output_stl_directory, output_pointcloud_directory):
        try:
            with h5py.File(stl_path.as_absolute(), 'r') as fp:
                # Grab stl
                Path(output_stl_directory + stl_path.subfolder_path).mkdir(parents=True, exist_ok=True)
                out_file = f"{output_stl_directory}/{stl_path.as_relative(extension=False)}.stl"

                # Now grab pointcloud by sampling from mesh
                cloud = load_stl_as_cloud(out_file)
                if cloud is None:
                    return ERROR_STATUS

                Path(output_pointcloud_directory + stl_path.subfolder_path).mkdir(parents=True, exist_ok=True)
                out_file = f"{output_pointcloud_directory}/{stl_path.as_relative(extension=False)}"
                np.save(out_file, cloud)

        except Exception as e:
            # print('cannot create stl:', e)
            return ERROR_STATUS

    multiproc = MultiProcQueueProcessing(args_global=(input_stl_directory, output_pointcloud_directory), task=task, num_workers=16)
    multiproc.process(process_list=stl_paths)

import argparse
if __name__ == "__main__":
    # input: data_path/name/stls
    # output: data_path/name/clouds
    parser = argparse.ArgumentParser(description="Convert STL to clouds.")
    parser.add_argument("-name", "--dataset_name", type=str, required=True, help="name of cad dataset")
    args = parser.parse_args()

    data_root = paths.DATA_PATH + args.dataset_name + "/"

    input_stl_dir = data_root + "/stls/"

    output_pc_dir = data_root + "/clouds/"

    Path(input_stl_dir).mkdir(parents=True, exist_ok=True)
    Path(output_pc_dir).mkdir(parents=True, exist_ok=True)

    print("Outputting to:")
    print(input_stl_dir)
    print(output_pc_dir)

    stl2pc(input_stl_dir, output_pc_dir)


    print("#" * 50)
    print("         PC creation completed        ")
    print("#" * 50)
