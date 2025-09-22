import trimesh
import trimesh_util
import numpy as np
import paths

if __name__=="__main__":
    data_dir = paths.HOME_PATH + "results/exp_name/autoencoder/reconstructions/clouds/"
    directory_manager = paths.DirectoryPathManager(data_dir, base_unit_is_file=True)
    files = directory_manager.get_files_absolute()

    for file in files:
        vec = np.load(file)
        values = np.zeros(len(vec))
        trimesh_util.show_mesh_with_normals(mesh=None, points=vec[:, :3], normals=vec[:, 3:])
