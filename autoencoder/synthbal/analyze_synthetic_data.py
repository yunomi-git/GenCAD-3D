
import numpy as np

import paths
from geometry.geometry_data import GeometryLoader
from pathlib import Path
import random
import h5py
from autoencoder.synthbal.augmentations import Augmentations
from cadlib.visualize import vec2CADsolid
from utils.visualizer_util import DaVinciVisualizer
import pyvista as pv
from OCC.Extend.DataExchange import write_stl_file
import trimesh

def save(dataset_dir, data_id, data):
    save_path = dataset_dir + data_id + ".h5"
    sub_dir = data_id[:data_id.find("/")]
    Path(dataset_dir + sub_dir).mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset('vec', data=data)


def augment(cad_vec, augmentation_manager):
    # augment on unpadded, concatenated cad_vec
    augmentations = {
        'replace_sketch': 0.6,
        # 'noise': 0.4,
    }

    augment_type = random.choices(list(augmentations.keys()), k=1, cum_weights=list(augmentations.values()))[0]

    if augment_type != 'noise':
        cad_vec = augmentation_manager.dataset_augment(cad_vec, "small_noise", aug_prob=0.0)  # 0 means full chance of augmentation

    cad_vec = augmentation_manager.dataset_augment(cad_vec, augment_type, aug_prob=0.0) # 0 means full chance of augmentation

    return cad_vec

test_id = "0009/00094598"

if __name__=="__main__":
    total_desired_cad_models = 170000

    data_root = paths.DATA_PATH + "GenCAD3D/"

    geometry_loader = GeometryLoader(data_root=data_root, phase=None, num_geometries=700)
    geometry_loader.get_all_valid_data(cad=True, mesh=False, cloud=False)


    num_cad_per_seq_length = total_desired_cad_models // 57

    existing_data_ids = geometry_loader.get_cad_ids_of_sequence_length(min_length=30, max_length=60,
                                                                       cad=True, mesh=False, cloud=False)

    num_augmentations_needed = 15
    visualizer = DaVinciVisualizer(geometry_loader)
    augmentation_manager = Augmentations(data_root, phase="train")


    while num_augmentations_needed > 0:
        # data_id = random.choice(existing_data_ids)
        data_id = test_id
        command, args = geometry_loader.load_cad(data_id, tensor=False, pad=False)
        cad_vec = np.hstack([command[:, np.newaxis], args])

        augmented_cad_vec = augment(cad_vec, augmentation_manager)
        print(data_id)

        try:
            shape = vec2CADsolid(augmented_cad_vec, return_validity=True)
            if shape is None:
                print("not valid")
                continue
            write_stl_file(shape, "temp.stl",
                           mode="binary",
                           linear_deflection=0.001,
                           angular_deflection=0.1)
            mesh = trimesh.load("temp.stl")
        except Exception as e:
            print("not valid", e)

            continue

        # if not valid:
        #     print("not valid")
        #     # continue

        # then plot
        default_size = 600
        pl = pv.Plotter(shape=(1, 2), window_size=[2 * default_size, default_size])
        pl.subplot(0, 0)
        visualizer.plot_picture(pl=pl, data_id=data_id, style="cad")
        pl.subplot(0, 1)
        visualizer.plot_picture(pl=pl, verts=mesh.vertices, faces=mesh.faces, style="cad")
        pl.link_views()
        pl.show()

        num_augmentations_needed -= 1

