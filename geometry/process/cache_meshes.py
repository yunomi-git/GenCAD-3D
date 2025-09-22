# This takes STLs and saves their faces and vertices
from pathlib import Path
import os
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

import paths
from paths import DirectoryPathManager
from .process_geometry import load_stl_as_mesh

def save_and_cache(my_task_id, num_tasks, input_folder, output_folder):
    master_path = paths.DATA_PATH + input_folder #"DaVinci_CAD_Augmented_NA_copy/remeshed_stls/"
    out_path_base = paths.DATA_PATH + output_folder # "DaVinci_CAD_Augmented_NA_copy/meshes/"

    Path(out_path_base).mkdir(parents=True, exist_ok=True)
    folders = os.listdir(master_path)
    folders.sort()
    # There are too many folders, so instead of doing all folders simultaneously, do them one at a time
    for folder in tqdm(folders):
        root_path = master_path + folder + "/"
        face_save_path = out_path_base + "faces/" + folder + "/"
        vertex_save_path = out_path_base + "verts/" + folder + "/"
        edge_save_path = out_path_base + "edges/" + folder + "/"

        Path(face_save_path).mkdir(parents=True, exist_ok=True)
        Path(edge_save_path).mkdir(parents=True, exist_ok=True)
        Path(vertex_save_path).mkdir(parents=True, exist_ok=True)

        directory_manager = DirectoryPathManager(root_path, base_unit_is_file=True)

        files = directory_manager.file_paths
        num_files = len(files)
        for i in range(my_task_id, num_files, num_tasks):
            file = files[i]

            # First check if it exists already
            if os.path.exists(face_save_path + file.as_relative(extension=False) + ".npy"):
                continue

            vertices, edges, faces = load_stl_as_mesh(file.as_absolute())

            np.save(face_save_path + file.as_relative(extension=False), faces)
            np.save(vertex_save_path + file.as_relative(extension=False), vertices)
            np.save(edge_save_path + file.as_relative(extension=False), edges)

def multiproc_remesh(num_workers, input_folder, output_folder):
    args_list = [(i, num_workers, input_folder, output_folder) for i in range(num_workers)]

    with Pool() as pool:
        pool.starmap(save_and_cache, args_list)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Convert an input directory of STLs to cached meshes")
    parser.add_argument("-input", "--input_folder", type=str, required=True, help="input directory of stls")
    parser.add_argument("-output", "--output_folder", type=str, required=True, help="output directory of cached meshes") # by default call this "meshes/"
    args = parser.parse_args()

    # Local preprocessing
    num_workers = os.cpu_count() - 2
    multiproc_remesh(num_workers, args.input_folder, args.output_folder)
