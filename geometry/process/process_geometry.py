# This file contains code to process a given geometry
import trimesh
from .. import trimesh_util
import numpy as np
from .remeshing import full_remesh_and_save

def load_stl_as_cloud(stl_file):
    # input: stl file
    # output: 4096 x 6 point cloud (xyz, normals)

    mesh = trimesh.load(stl_file)

    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    num_desired_vertices = 4096
    sampled_values, normals = mesh_aux.sample_and_get_normals(count=num_desired_vertices * 2, use_weight="even")

    sampled_values = sampled_values[:num_desired_vertices]
    normals = normals[:num_desired_vertices]
    augmented_samples = np.concatenate((sampled_values, normals), axis=1)
    if len(augmented_samples) < num_desired_vertices:
        return None

    return augmented_samples

def remesh_stl(input_stl_path, output_stl_path):
    full_remesh_and_save(input_stl_path, output_stl_path, min_vertices=2048, max_vertices=4000)

def load_stl_as_mesh(stl_file):
    try:
        mesh = trimesh.load(stl_file)
        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)
    except:
        print("error loading trimesh", stl_file)
        return

    if not mesh_aux.is_valid:
        return

    if not mesh.is_watertight:
        print("not watertight: ", stl_file)
        return

    faces = mesh.faces.astype(np.int64)
    edges = mesh.edges.astype(np.int64)
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh_aux.vertex_normals.astype(np.float32)

    # Attach normals to vertices
    vertices = np.concatenate([vertices, normals], axis=1)

    if np.max(faces) > len(vertices):
        print("faces > vertices")
        return
    if np.max(edges) > len(vertices):
        print("edges > vertices")
        return

    return vertices, edges, faces
