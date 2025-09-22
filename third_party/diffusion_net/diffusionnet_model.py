from __future__ import annotations

import numpy as np
import torch.nn as nn
import torch
from .layers import DiffusionNet
from . import utils
from . import geometry

def mesh_is_valid_diffusion(verts, faces):
    # Invalid if a face index is larger than the number of verticies
    if torch.max(faces) > len(verts):
        return False
    # verts_np = toNP(verts).astype(np.float32)
    # faces_np = toNP(faces)
    verts_np, faces_np = utils.mesh_to_np(verts, faces)
    return not np.max(faces_np) > len(verts_np)

def mesh_has_valid_operators(tensor_vert, tensor_face, k_eig, op_cache_dir):
    try:
        geometry.get_operators(tensor_vert, tensor_face, k_eig=k_eig, op_cache_dir=op_cache_dir)
    except Exception as e:  # Or valueerror or ArpackError
        print("Error calculating decomposition. Skipping:", e)
        return False
    return True

def translate_pointcloud(pointcloud, device):
    # pointcloud is points x dim
    xyz_add = torch.zeros(pointcloud.size()[-1])
    xyz_add[:3] = torch.from_numpy(np.random.uniform(low=-0.1, high=0.1, size=[3]))

    return pointcloud + xyz_add.to(device)


class DiffusionNetWrapper(nn.Module):
    def __init__(self, model_args, op_cache_dir, device, augment_perturb_position=False):
        super(DiffusionNetWrapper, self).__init__()

        input_feature_type = model_args["input_feature_type"]
        num_outputs = model_args["num_outputs"]
        C_width = model_args["C_width"]
        N_block = model_args["N_block"]
        # last_activation = model_args["last_activation"]
        self.outputs_at = model_args["outputs_at"]
        output_at_to_pass = self.outputs_at
        if output_at_to_pass == "global":
            output_at_to_pass = "global_mean"
        mlp_hidden_dims = model_args["mlp_hidden_dims"]
        dropout = model_args["dropout"]
        with_gradient_features = model_args["with_gradient_features"]
        with_gradient_rotations = model_args["with_gradient_rotations"]
        diffusion_method = model_args["diffusion_method"]
        self.k_eig = model_args["k_eig"]
        self.device = device

        self.last_layer = None
        if "last_layer" in model_args:
            if model_args["last_layer"] == "sigmoid":
                self.last_layer = torch.nn.Sigmoid()
            elif model_args["last_layer"] == "softmax":
                self.last_layer = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.LogSoftmax(dim=-1))


        self.input_feature_type = input_feature_type
        self.op_cache_dir = op_cache_dir
        C_in = {'xyz': 3, 'hks': 16}[self.input_feature_type]

        self.augment_perturb_position = augment_perturb_position

        self.wrapped_model = DiffusionNet(C_in=C_in, C_out=num_outputs, C_width=C_width, N_block=N_block,
                                          outputs_at=output_at_to_pass,
                                          mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                          with_gradient_features=with_gradient_features,
                                          with_gradient_rotations=with_gradient_rotations,
                                          last_activation=self.last_layer,
                                          diffusion_method=diffusion_method)

    def forward(self, verts, faces,
                folder=None,
                frames=None, mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None):
        # TODO: this assumes batch size 1 right now
        # Calculate properties
        verts = verts[0]
        faces = faces[0]
        if frames is not None:
            frames = frames[0]
            mass = mass[0]
            L = L[0]
            evals = evals[0]
            evecs = evecs[0]
            gradX = gradX[0]
            gradY = gradY[0]
        else:
            op_cache_dir = self.op_cache_dir
            if folder is not None:
                folder = folder[0]
                op_cache_dir += folder
            frames, mass, L, evals, evecs, gradX, gradY = geometry.get_operators(verts[:, :3], faces,
                                                                                       k_eig=self.k_eig,
                                                                                       op_cache_dir=op_cache_dir)
        if self.augment_perturb_position:
            verts = translate_pointcloud(verts, self.device)

        verts = verts.to(self.device)
        faces = faces.to(self.device)
        # frames = frames.to(device)
        mass = mass.to(self.device)
        L = L.to(self.device)
        evals = evals.to(self.device)
        evecs = evecs.to(self.device)
        gradX = gradX.to(self.device)
        gradY = gradY.to(self.device)
        # Construct features
        if self.input_feature_type == 'xyz':
            features = verts
        else:  # self.input_feature_type == 'hks':
            features = geometry.compute_hks_autoscale(evals, evecs, 16)  # TODO autoscale here
            # TODO Append extra labels here if chosen to do so

        out = self.wrapped_model.forward(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX,
                                          gradY=gradY, faces=faces)
        out = torch.nan_to_num(out) # TODO is this the best option?
        if self.outputs_at == "vertices":
            return out[None, :, :]
        else:
            return out[None, :]