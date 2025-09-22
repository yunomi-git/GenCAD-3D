# ----------------------------
# 
# Code adapted from: https://github.com/ChrisWu1997/DeepCAD
#
#-----------------------------
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
from cadlib.macro import EOS_VEC
import numpy as np
import random
from multiprocessing import cpu_count
from PIL import Image
from torchvision import transforms
import cv2
from third_party.diffusion_net import utils
import potpourri3d as pp3d
from geometry import trimesh_util
from cadlib.util import load_cad_vec

def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_contrastive_cloud_dataloader(phase, num_points, with_normals, batch_size, data_root, shuffle=None, num_workers=None, virtual_scan=False, normalize=False):
    dataset = CCIPCloudDataset(phase, data_root=data_root, with_normals=with_normals, num_points=num_points, normalize=normalize)
    dataloader = create_dataloader(dataset, phase, batch_size, shuffle, num_workers)

    return dataloader

def get_contrastive_mesh_feast_dataloader(phase, mesh_folder, use_normals, data_root, batch_size, shuffle=None, num_workers=None, check_valid_mesh=True):
    dataset = CCIPMeshFeastDataset(phase, data_root, mesh_folder, with_normals=use_normals, check_valid_mesh=check_valid_mesh)
    dataloader = create_dataloader(dataset, phase, batch_size, shuffle, num_workers)
    return dataloader    



def create_dataloader(dataset, phase, batch_size, shuffle, num_workers, collate_fn=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle


    if num_workers is None:
        num_workers = cpu_count()
    num_workers = min(num_workers, cpu_count())

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_shuffle, num_workers=num_workers,
                            pin_memory = True, collate_fn=collate_fn,
                            drop_last=True)
    return dataloader


class CCIPDataset(Dataset):
    def __init__(self, phase, config, img_type="cad_image"):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.img_type = img_type
        self.path = os.path.join(config.data_root, "filtered_data.json")
        self.img_id_path = os.path.join(config.data_root, "image_ids.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with open(self.img_id_path, "r") as fp:
            self.img_ids = json.load(fp)

        if self.img_type == "cad_image":
            self.raw_image = os.path.join(config.data_root, "images")
        elif self.img_type == "cad_sketch":
            self.raw_image = os.path.join(config.data_root, "sketches")

        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5]),
                    ])


        self.max_total_len = 60
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        img_id_by_data_id = self.img_ids[data_id]   # [0, 1, 2, 3, 4]

        # randomly choose one of the image id
        img_id = random.choice(img_id_by_data_id)
        image_path = os.path.join(self.raw_image, data_id + "_" + str(img_id) + ".png")
                
        image = Image.open(image_path)
        gray_img = np.array(image.convert("L"))

        # make edges more clear
        edges = cv2.Canny(gray_img, 50, 150) 
        kernel = np.ones((3, 3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)

        if self.img_type == "cad_sketch":

            # Convert to 3-channel image
            edges_3channel = np.stack([thick_edges]*3, axis=-1)
            edges_pil = Image.fromarray(edges_3channel)
            image_tensor = self.preprocess(edges_pil)

        else:
            # enhanced_img = cv2.addWeighted(gray_img, 1.0, thick_edges, 0.5, 0)
            # enhanced_img_3channel = np.stack([enhanced_img]*3, axis=-1)
            # enhanced_img_pil = Image.fromarray(enhanced_img_3channel)
            # image_tensor = self.preprocess(enhanced_img_pil)
            rgb_img = image.convert("RGB")  # remove transparent background
            image_tensor = self.preprocess(rgb_img)


        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "image":image_tensor, "id": data_id}

    def __len__(self):
        return len(self.all_data)
    



class CCIPCloudDataset(Dataset):
    def __init__(self, phase, with_normals, num_points, data_root, normalize=False, noisy_normals=False):
        super().__init__()

        self.raw_data = os.path.join(data_root, "cad_vec")  # h5 data root
        self.phase = phase
        self.path = os.path.join(data_root, "filtered_data.json")

        with open(self.path, "r") as fp:
            all_data = json.load(fp)[phase]

        self.raw_mesh = os.path.join(data_root, "clouds")

        # Check if cloud exists
        # Check if model loads correctly
        # # check if data is valid:
        self.all_data = []
        for data_id in tqdm(all_data):
            h5_path = os.path.join(self.raw_data, data_id + ".h5")
            cloud_path = os.path.join(self.raw_mesh, data_id + ".npy")
            if not os.path.exists(h5_path):
                continue
            if not os.path.exists(cloud_path):
                continue

            self.all_data.append(data_id)

        print("Num issues", len(all_data) - len(self.all_data))

        self.normalize = normalize
        if self.normalize:
            print("Normalizing dataset")
        self.with_normals = with_normals
        self.num_points = num_points
        self.noisy_normals = noisy_normals

        self.max_total_len = 60
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]

        # Grab Cloud
        cloud_path = os.path.join(self.raw_mesh, data_id + ".npy")
        pointcloud = np.load(cloud_path)
        if not self.with_normals:
            pointcloud = pointcloud[:, :3]

        elif self.noisy_normals:
            strength=0.1
            normals = pointcloud[:, 3:]
            noise = np.random.normal(0, strength, normals.shape)
            noisy_normals = normals + noise
            noisy_normals /= np.linalg.norm(noisy_normals, axis=1, keepdims=True)
            pointcloud[:, 3:] = noisy_normals

        if self.normalize:
            pointcloud[:, :3] = trimesh_util.normalize_vertices(pointcloud[:, :3], center=True, scale=True)

        # Grab random num_points from the point cloud
        if self.phase == 'train':
            p = np.random.permutation(len(pointcloud))[:self.num_points]
        else:
            p = np.arange(self.num_points)

        pointcloud = pointcloud[p]
        # TODO Apply random transformation (rotation or translation) if desired

        # Grab Cad vec
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]  # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        pointcloud = torch.tensor(pointcloud, dtype=torch.float)
        return {"command": command, "args": args, "image": pointcloud, "id": data_id}

    def __len__(self):
        return len(self.all_data)


class CCIPMeshFeastDataset(Dataset):
    def __init__(self, phase, data_root, mesh_folder="meshes/", with_normals=False, check_valid_mesh=True):
        super().__init__()
        # For CAD
        self.max_total_len = 60
        self.size = 256

        self.raw_data = os.path.join(data_root, "cad_vec/")  # h5 data root
        self.phase = phase
        self.path = os.path.join(data_root, "filtered_data.json")
        with open(self.path, "r") as fp:
            all_data = json.load(fp)[phase]

        self.raw_vert_dir = os.path.join(data_root, mesh_folder + "verts/")
        self.raw_edge_dir = os.path.join(data_root, mesh_folder + "edges/")
        self.raw_face_dir = os.path.join(data_root, mesh_folder + "faces/")

        # Check if cloud exists
        # Check if model loads correctly
        # # check if data is valid:
        self.all_data = []
        self.all_verts = []
        self.all_faces = []
        self.all_cad = []

        self.with_normals = with_normals

        for data_id in tqdm(all_data):
            h5_path = os.path.join(self.raw_data, data_id + ".h5")
            cloud_path = os.path.join(self.raw_vert_dir, data_id + ".npy")
            edge_path = os.path.join(self.raw_edge_dir, data_id + ".npy")
            # face_path = os.path.join(self.raw_face_dir, data_id + ".npy")
            if not os.path.exists(h5_path):
                # print("no cad")
                continue
            if not os.path.exists(cloud_path):
                # print("no verts")
                continue
            if not os.path.exists(edge_path):
                # print("no edge")
                continue
            # if not os.path.exists(face_path):
                # print("no face")
                # continue
            if check_valid_mesh:
                try:
                    verts, edges, faces = self.load_verts_faces(data_id)
                except Exception:
                    print("load error", data_id)
                    continue
                # if np.max(faces) > len(verts):
                #     print("face index error", data_id)
                #     continue
                if np.max(edges) > len(verts):
                    print("edge index error", data_id)
                    continue

            self.all_data.append(data_id)

        print("Num issues", len(all_data) - len(self.all_data))

    def load_verts_faces(self, data_id):
        vert_path = os.path.join(self.raw_vert_dir, data_id + ".npy")
        face_path = os.path.join(self.raw_face_dir, data_id + ".npy")
        edge_path = os.path.join(self.raw_edge_dir, data_id + ".npy")

        verts = np.load(vert_path).astype(np.float32)
        if not self.with_normals:
            verts = verts[..., :3]
        edges = np.load(edge_path).astype(np.int64)
        faces = np.load(face_path).astype(np.int64)

        return verts, edges, faces

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]

        # Grab Cloud
        verts, edges, faces = self.load_verts_faces(data_id)

        eps = 1e-8
        try:
            mass = pp3d.vertex_areas(verts[..., :3], faces)
            mass += eps * np.mean(mass)
            mass = mass.astype(np.float32)
        except Exception:
            mass = np.ones(verts.shape[0]).astype(np.float32)
            print("mass load error")
        # Grab Cad vec
        command, args = load_cad_vec(data_path=self.raw_data, data_id=data_id, max_total_len=self.max_total_len)
        verts, edges = utils.mesh_to_tensor(verts, edges)

        return {"command": command, "args": args,
                "image": (verts, edges.T, mass),
                "id": data_id}


    def __len__(self):
        return len(self.all_data)