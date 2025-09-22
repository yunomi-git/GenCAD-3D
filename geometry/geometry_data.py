import os
import numpy as np
import json

import trimesh
from tqdm import tqdm
import torch
from autoencoder.cad_dataset import load_cad_vec
import potpourri3d as pp3d
from cadlib.macro import (
    EOS_IDX
)
from geometry import trimesh_util

def calculate_mass(verts, faces):
    eps = 1e-8
    verts = verts[..., :3]
    try:
        mass = pp3d.vertex_areas(verts, faces)
        mass += eps * np.mean(mass)
        mass = mass.astype(np.float32)
    except Exception:
        mass = np.ones(verts.shape[0]).astype(np.float32)
        print("mass load error")
    return mass


class GeometryLoader:
    def __init__(self, data_root, phase="train", geometry_subdir="meshes/", with_normals=False, stl_directory="stls/", num_geometries=None, splits_file="filtered_data", split_cad=False):
        # For CAD
        self.max_total_len = 60
        self.size = 256
        self.with_normals = with_normals

        self.num_geometries = num_geometries

        self.data_root = data_root

        self.cad_dir = os.path.join(data_root, "cad_vec/")  # h5 data root
        self.phase = phase

        self.splits_file = os.path.join(data_root, splits_file + ".json")

        self.split_cad = split_cad

        if geometry_subdir is not None:
            self.vert_dir = os.path.join(data_root, geometry_subdir + "verts/")
            self.edge_dir = os.path.join(data_root, geometry_subdir + "edges/")
            self.face_dir = os.path.join(data_root, geometry_subdir + "faces/")
        self.cloud_dir = os.path.join(data_root, "clouds/")

        self.stl_dir = os.path.join(data_root, stl_directory)
        self.step_dir = os.path.join(data_root, "step/")

        self.all_valid_data_cache = {}

        self.sequence_length_cache = None

    def standard_modality_checks(self, encoder_type):
        if "mesh" in encoder_type:
            return {"mesh": True, "cad": True}
        else:
            return {"cloud": True, "cad": True}
    

    def get_all_valid_data(self, cad=False, cloud=False, mesh=False, stl=False):
        cache_key = (cad, cloud, mesh, stl)
        if cache_key not in self.all_valid_data_cache:
            # Load from splits
            if self.splits_file is not None:
                with open(self.splits_file, "r") as fp:
                    if self.phase is not None:
                        all_data = json.load(fp)[self.phase]
                    else:
                        all_data = []
                        dataset = json.load(fp)
                        for phase in ["train", "test", "validation"]:
                            all_data += dataset[phase]

            self.all_valid_data_cache[cache_key] = []

            print("Checking all data for validity")
            for data_id in tqdm(all_data[:self.num_geometries]): # Note - [:None] == [:]
                if self.check_valid_data(data_id, cloud=cloud, mesh=mesh, cad=cad, stl=stl):
                    self.all_valid_data_cache[cache_key].append(data_id)

            if self.num_geometries is not None:
                self.all_valid_data_cache = self.all_valid_data_cache

            print("Num issues", len(all_data) - len(self.all_valid_data_cache[cache_key]))
            print("Num data loaded: ", len(self.all_valid_data_cache[cache_key]))

        return self.all_valid_data_cache[cache_key]

    def check_valid_data(self, data_id, cad=False, cloud=False, mesh=False, stl=False):
        h5_path = os.path.join(self.cad_dir, data_id + ".h5")
        if cad and not os.path.exists(h5_path):
            return False

        cloud_path = os.path.join(self.cloud_dir, data_id + ".npy")
        if cloud and not os.path.exists(cloud_path):
            return False

        if mesh:
            verts_path = os.path.join(self.vert_dir, data_id + ".npy")
            edge_path = os.path.join(self.edge_dir, data_id + ".npy")
            # face_path = os.path.join(self.face_dir, data_id + ".npy")
            if not os.path.exists(verts_path):
                return False
            if not os.path.exists(edge_path):
                return False
            # if not os.path.exists(face_path):
            #     return False

        if stl:
            stls_path = os.path.join(self.stl_dir, data_id + ".stl")
            if not os.path.exists(stls_path):
                return False

        return True

    def load_stl_vef(self, data_id=None, stl_path=None, as_tensor=False, device="cuda"):
        assert not data_id is None and stl_path is None
        if stl_path is None:
            stl_path = os.path.join(self.stl_dir, data_id + ".stl")
        mesh = trimesh.load(stl_path)
        verts = mesh.vertices
        faces = mesh.faces
        edges = mesh.edges

        if as_tensor:
            verts = verts[np.newaxis, ...]  # batch size 1
            verts = torch.tensor(verts, dtype=torch.float32).to(device)
            edges = edges[np.newaxis, ...]  # batch size 1
            edges = torch.tensor(edges, dtype=torch.int64).to(device)
            faces = faces[np.newaxis, ...]  # batch size 1
            faces = torch.tensor(faces, dtype=torch.int64).to(device)

        return verts, edges, faces

    def load_mesh_vef(self, data_id, as_tensor=False, device="cuda", normalize=False):
        if not self.check_valid_data(data_id, mesh=True):
            print("Mesh not found:", data_id)
            return
        vert_path = os.path.join(self.vert_dir, data_id + ".npy")
        face_path = os.path.join(self.face_dir, data_id + ".npy")
        edge_path = os.path.join(self.edge_dir, data_id + ".npy")

        verts = np.load(vert_path).astype(np.float32)
        if not self.with_normals:
            verts = verts[..., :3]
        edges = np.load(edge_path).astype(np.int64)
        faces = np.load(face_path).astype(np.int64)

        if normalize:
            verts[:, :3] = trimesh_util.normalize_vertices(verts[:, :3], center=True, scale=True)

        if as_tensor:
            verts = verts[np.newaxis, ...]  # batch size 1
            verts = torch.tensor(verts, dtype=torch.float32).to(device)
            edges = edges[np.newaxis, ...]  # batch size 1
            edges = torch.tensor(edges, dtype=torch.int64).to(device)
            faces = faces[np.newaxis, ...]  # batch size 1
            faces = torch.tensor(faces, dtype=torch.int64).to(device)

        return verts, edges, faces

    def load_mesh_vef_batch(self, data_ids):
        # Returns list of tensors data x [1 x items x info]
        verts_list = []
        edges_list = []
        faces_list = []

        for data_id in data_ids:
            verts, edges, faces = self.load_mesh_vef(data_id, as_tensor=True)
            verts_list.append(verts)
            edges_list.append(edges)
            faces_list.append(faces)

        return verts_list, edges_list, faces_list

    def load_cloud(self, data_id, num_points=2048, as_tensor=False, device="cuda", normalize=False):
        if not self.check_valid_data(data_id, cloud=True):
            return

        cloud = np.load(self.cloud_dir + data_id + ".npy")

        if normalize:
            cloud[:, :3] = trimesh_util.normalize_vertices(cloud[:, :3], center=True, scale=True)

        if not self.with_normals:
            cloud = cloud[..., :3]
        cloud = cloud[:num_points]
        if as_tensor:
            cloud = cloud[np.newaxis, ...]  # batch size 1
            cloud = torch.tensor(cloud, dtype=torch.float32).to(device)
        return cloud

    def load_cloud_batch(self, data_ids, device="cuda", scan=False, normalize=False):
        # returns tensor batch [data x items x info]
        clouds = []
        for cad_id in data_ids:
            cloud = self.load_cloud(cad_id, as_tensor=False, normalize=normalize)
            clouds.append(cloud)
        clouds = np.stack(clouds, axis=0)
        clouds = torch.tensor(clouds, dtype=torch.float).to(device)
        return clouds

    def load_cad(self, data_id, tensor=True, pad=True, as_single_vec=False):
        # always a tensor
        # command, args = load_cad_vec(data_path=self.cad_dir, data_id=data_id, max_total_len=self.max_total_len, pad=pad, tensor=tensor)
        # return command, args
        return load_cad_vec(data_path=self.cad_dir, data_id=data_id, max_total_len=self.max_total_len, pad=pad, as_tensor=tensor, as_single_vec=as_single_vec)

    def load_cad_batch(self, data_ids, device="cuda", as_single_vec=False):
        if not as_single_vec:
            command_list = []
            arg_list = []
            for cad_id in data_ids:
                command, arg = self.load_cad(cad_id, as_single_vec=as_single_vec)
                command_list.append(command)
                arg_list.append(arg)
            command_list = torch.stack(command_list, dim=0).to(device)
            arg_list = torch.stack(arg_list, dim=0).to(device)
            return command_list, arg_list
        else:
            cad_list = []
            for cad_id in data_ids:
                cad_vec = self.load_cad(cad_id, as_single_vec=as_single_vec)
                cad_list.append(cad_vec)
            cad_list = torch.stack(cad_list, dim=0).to(device)
            return cad_list

    def calc_cad_sequence_length(self, data_id):
        command, args = self.load_cad(data_id)
        seq_len = command.tolist().index(EOS_IDX)
        return seq_len

    def get_cad_ids_of_sequence_length(self, min_length=None, max_length=None, cloud=False, mesh=False, cad=True, scan=False):
        if self.sequence_length_cache is None:
            all_valid_ids = self.get_all_valid_data(cloud=cloud, mesh=mesh, cad=cad)
            self.sequence_length_cache = {}
            print("Caching Sequence Lengths")
            for cad_id in tqdm(all_valid_ids):
                try:
                    self.sequence_length_cache[cad_id] = self.calc_cad_sequence_length(cad_id)
                except Exception as e:
                    print("ID not found", cad_id)

        # Now grab them as a list of ids
        relevant_ids = []
        for cad_id in self.sequence_length_cache.keys():
            seq_len = self.sequence_length_cache[cad_id]
            if (min_length is not None and seq_len < min_length):
                continue
            if (max_length is not None and seq_len > max_length):
                continue

            relevant_ids.append(cad_id)

        return relevant_ids