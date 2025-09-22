import geometry.trimesh_util as trimesh_util
import pymeshlab
import trimesh

def trimesh_to_pymesh(mesh: trimesh.Trimesh):
    temp_file_name = "temp.stl"
    mesh.export(temp_file_name, "stl")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_file_name)
    return ms

def pymesh_to_trimesh(ms):
    temp_file_name = "temp.stl"
    ms.save_current_mesh(temp_file_name,
                         save_face_color=False)
    mesh = trimesh.load(temp_file_name)
    return mesh