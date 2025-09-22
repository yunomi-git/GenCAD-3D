import os.path

import numpy as np
import trimesh

from geometry.geometry_data import GeometryLoader
from utils.util import context_print
from geometry import trimesh_util
from third_party.FeaStNet.models import FeaStNet
import torch
import paths
import json
from autoencoder.model_utils import logits2vec
from cadlib.macro import (MAX_TOTAL_LEN)
from cadlib.macro import SOL_IDX
from autoencoder.gencad.model import VanillaCADTransformer
from autoencoder.configAE import ConfigAE
from contrastive.model import CLIP, DavinciClipAdapter, l2norm, XClipAdapter
from contrastive.datasets import get_contrastive_cloud_dataloader, get_contrastive_mesh_feast_dataloader
from contrastive.dgcnn_model import DGCNN_param
from pathlib import Path
import pickle
from tqdm import tqdm
from contrastive.configContrastive import ContrastivePathConfig

DEFAULT_ENCODER_LOAD_ARGS = {
    "pcn": {
        "batch_size": 128
    },
    "pc": {
        "batch_size": 16
    },
    "mesh_feast": {
        "batch_size": 1
    },
}


def geo_numpy_to_tensor(vertices, edges=None, device="cuda"):
    verts = vertices[np.newaxis, ...]  # batch size 1
    verts = torch.tensor(verts, dtype=torch.float32).to(device)

    if edges is not None:
        edges = edges[np.newaxis, ...]  # batch size 1
        edges = torch.tensor(edges, dtype=torch.int64).to(device)
        return (verts, torch.transpose(edges, 1, 2))
    else:
        return (verts, )

def load_from_stl(stl_path, encoder_type, use_normals, num_points=2048):
    mesh = trimesh.load(stl_path)
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

    if "mesh" in encoder_type:
        vertices = mesh_aux.vertices
        normals = mesh_aux.vertex_normals
        edges = mesh_aux.edges
        faces = mesh_aux.faces
        if use_normals:
            vertices = np.hstack([vertices, normals])
        return vertices, edges, faces

    elif encoder_type == "pc" or encoder_type == "pcn":
        cloud, normals = mesh_aux.sample_and_get_normals(count=num_points)
        if use_normals:
            cloud = np.hstack([cloud, normals])

        return cloud

def load_encoder_input_from_stl(stl_path, encoder_type, use_normals, device="cuda"):
    loaded_stl = load_from_stl(stl_path, encoder_type, use_normals)

    if "mesh" in encoder_type:
        vertices = loaded_stl[0]
        edges = loaded_stl[1]
        return geo_numpy_to_tensor(vertices, edges, device=device)

    elif encoder_type == "pc" or encoder_type == "pcn":
        cloud = loaded_stl
        return geo_numpy_to_tensor(cloud, device=device)

class GeometryEmbeddingSpace:
    def __init__(self, cache_parent_dir, encoder_type, num_geometries, geometry_loader: GeometryLoader, ckpt_name, generation=False, dataset_name=None, min_seq_len=None, max_seq_len=None, normalize=False):
        # This applies to the test set
        # Genreates a cache of all embeddings.
        # if a smaller number of embeddings is requested, grab them from the full cache
        # set num_geometries = None to grab all geometries in the set

        self.phase = geometry_loader.phase
        self.cache_folder = cache_parent_dir #+ "/" + encoder_type + str(num_geometries) + "/"
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
        self.encoder_type = encoder_type
        self.num_geometries = num_geometries

        self.normalize = normalize


        self.ckpt_name = ckpt_name
        self.cad_ids_folder = self.cache_folder + "cad_ids/"
        self.geometries_folder = self.cache_folder + self.encoder_type + "/"
        self.cad_folder = self.cache_folder + "cad/"
        Path(self.cad_ids_folder).mkdir(parents=True, exist_ok=True)
        Path(self.geometries_folder).mkdir(parents=True, exist_ok=True)
        Path(self.cad_folder).mkdir(parents=True, exist_ok=True)

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.generation = generation
        if generation:
            save_postfix = "gen"
        else:
            save_postfix = "retr"

        if self.normalize:
            save_postfix += "_normalize"

        if dataset_name is None:
            dataset_name = ""
        else:
            dataset_name = "_" + dataset_name


        self.get_geometry_save_name = lambda num_geo: self.geometries_folder + ckpt_name + "_" + save_postfix + str(num_geo) + dataset_name + f"-{self.phase}.pickle"
        self.get_cad_id_save_name = lambda num_geo: self.cad_ids_folder + "ids" + str(num_geo) + dataset_name + f"-{self.phase}.pickle"
        self.get_cad_save_name = lambda num_geo: self.geometries_folder + "cad_emb_" + ckpt_name + "_" + save_postfix + str(num_geo) + dataset_name + f"-{self.phase}.pickle"

        self.CACHE_SPACE_SIZE = "FULL_" + self.phase.upper() #7376

        self.geometry_loader = geometry_loader


        self.geometry_save_name = self.get_geometry_save_name(self.CACHE_SPACE_SIZE)
        self.cad_id_save_name = self.get_cad_id_save_name(self.CACHE_SPACE_SIZE)
        self.cad_emb_save_name = self.get_cad_save_name(self.CACHE_SPACE_SIZE)

        self.all_cached_ids = self.load_cached_space_cad_ids_with_seq_len(self.min_seq_len, self.max_seq_len)
        self.num_in_space = len(self.all_cached_ids)

        if self.num_geometries is None:
            self.num_geometries = self.num_in_space

        # when loading geometries from cad
        self.subbatch_indices = np.arange(self.num_geometries)

        self.cad_id_to_idx = {self.all_cached_ids[i]: i for i in range(self.num_in_space)}

        self.all_cached_geometry_embeddings = None
        self.all_cached_cad_embeddings = None

    def set_specific_subbatch(self, cad_ids):
        self.subbatch_indices = [self.cad_id_to_idx[cad_id] for cad_id in cad_ids]
        self.subbatch_indices = np.array(self.subbatch_indices)

    def do_randomize_subbatch(self):
        self.subbatch_indices = np.random.choice(np.arange(self.num_in_space), size=self.num_geometries, replace=False)

    def load_all_cached_space_cad_ids(self):
        space_cad_ids = self.geometry_loader.get_all_valid_data(**self.geometry_loader.standard_modality_checks(encoder_type=self.encoder_type))

        return space_cad_ids

    def load_cached_space_cad_ids_with_seq_len(self, min_length=None, max_length=None):
        if min_length is None and max_length is None:
            return self.load_all_cached_space_cad_ids()
        space_cad_ids = self.geometry_loader.get_cad_ids_of_sequence_length(min_length=min_length, max_length=max_length, **self.geometry_loader.standard_modality_checks(self.encoder_type))

        return space_cad_ids

    def load_space_cad_ids(self):
        # Loads self.num_geometries ids from cache of all space cad ids
        # If not existing creates a cache of self.CACHE_SPACE_SIZE cad ids
        # space_cad_ids = self.load_all_cached_space_cad_ids()
        space_cad_ids = self.load_cached_space_cad_ids_with_seq_len(self.min_seq_len, self.max_seq_len)

        return [space_cad_ids[idx] for idx in self.subbatch_indices]

    def load_all_cached_geometry_embeddings(self, geometry_to_cad, normalize=False):
        if not os.path.exists(self.get_geometry_save_name(self.CACHE_SPACE_SIZE)):
            print("Geometry Cache not found. Generating Cache:", self.get_geometry_save_name(self.CACHE_SPACE_SIZE))
            self.generate_full_geometry_embedding_space(geometry_to_cad=geometry_to_cad)

        if self.all_cached_geometry_embeddings is None:
            with open(self.get_geometry_save_name(self.CACHE_SPACE_SIZE), 'rb') as handle:
                self.all_cached_geometry_embeddings = pickle.load(handle)

        embeddings = self.all_cached_geometry_embeddings

        if normalize:
            embeddings = l2norm(embeddings)
        return embeddings

    # Save cad ids separately
    def load_geometry_space_embeddings(self, geometry_to_cad, normalize=False):
        space_cad_ids = self.load_space_cad_ids()
        embeddings = self.load_all_cached_geometry_embeddings(geometry_to_cad=geometry_to_cad, normalize=normalize)
        embeddings = embeddings[self.subbatch_indices]

        return embeddings, space_cad_ids

    def load_all_cached_cad_embeddings(self, geometry_to_cad, normalize=False):
        if not os.path.exists(self.cad_emb_save_name):
            print("CAD Cache not found. Generating Cache")
            self.generate_full_cad_space(geometry_to_cad=geometry_to_cad)

        if self.all_cached_cad_embeddings is None:
            with open(self.cad_emb_save_name, 'rb') as handle:
                self.all_cached_cad_embeddings = pickle.load(handle)

        embeddings = self.all_cached_cad_embeddings

        if normalize:
            embeddings = l2norm(embeddings)
        return embeddings

    def load_cad_embeddings(self, geometry_to_cad, normalize=False):
        space_cad_ids = self.load_space_cad_ids()

        embeddings = self.load_all_cached_cad_embeddings(geometry_to_cad, normalize=normalize)
        embeddings = embeddings[self.subbatch_indices]

        return embeddings, space_cad_ids

    def generate_full_geometry_embedding_space(self, geometry_to_cad):
        # Generate full cache of embeddings
        batch_size = DEFAULT_ENCODER_LOAD_ARGS[self.encoder_type]["batch_size"]
        space_cad_ids = self.load_all_cached_space_cad_ids() # Generate all and cache them
        space_size = len(space_cad_ids)
        embeddings = []
        # batch it up
        for j in tqdm(range(int(np.ceil(space_size / batch_size)))):
            batch_cad_ids = space_cad_ids[j * batch_size:(j + 1) * batch_size]
            try:
                batch = load_geometry_input(encoder_type=self.encoder_type,
                                            data_ids=batch_cad_ids,
                                            geometry_loader=self.geometry_loader,
                                            normalize=self.normalize)
            except Exception as e:
                context_print("load batch error: " + str(e))
                return # Error if embeddings is not found

            batch_embeddings = geometry_to_cad.encode_geometry(*batch, generation=self.generation)
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings)

        print("saving geometry cache to ", self.cache_folder)
        paths.mkdir(self.geometry_save_name)
        with open(self.geometry_save_name, 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

    def generate_full_cad_space(self, geometry_to_cad):
        # Only generates the requested cad ids. Does not cache
        batch_size = 32
        space_cad_ids = self.load_all_cached_space_cad_ids() # Generate all and cache them
        embeddings = []
        space_size = len(space_cad_ids)
        # batch it up
        for j in tqdm(range(int(np.ceil(space_size / batch_size)))):
            batch_cad_ids = space_cad_ids[j * batch_size:(j + 1) * batch_size]
            commands, args = self.load_cad_input(batch_cad_ids)
            batch_embeddings = geometry_to_cad.encode_cad(commands, args, generation=self.generation)
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings)

        print("saving cad cache to ", self.cache_folder)
        with open(self.cad_emb_save_name, 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

    def load_embeddings(self, cad_ids, geometry_to_cad):
        query_geometry_embeddings = []
        query_cad_embeddings = []

        all_geometry_embeddings = self.load_all_cached_geometry_embeddings(geometry_to_cad=geometry_to_cad)
        all_cad_embeddings = self.load_all_cached_cad_embeddings(geometry_to_cad=geometry_to_cad)
        all_cad_ids = self.load_all_cached_space_cad_ids()

        for cad_id in cad_ids:
            index = all_cad_ids.index(cad_id)
            query_geometry_embeddings.append(all_geometry_embeddings[index])
            query_cad_embeddings.append(all_cad_embeddings[index])

        return torch.stack(query_geometry_embeddings, dim=0), torch.stack(query_cad_embeddings, dim=0)

    def load_cad_input(self, data_ids):
        return self.geometry_loader.load_cad_batch(data_ids)


def load_geometry_input(encoder_type, data_ids, geometry_loader: GeometryLoader, normalize=False):
    if encoder_type == "pcn" or encoder_type == "pc":
        return (geometry_loader.load_cloud_batch(data_ids, scan=False, normalize=normalize),)
    elif encoder_type == "mesh_feast":
        # Enforce that batch size is 1
        assert len(data_ids) == 1
        verts, edges, _ = geometry_loader.load_mesh_vef(data_ids[0], as_tensor=True, normalize=normalize)
        return (verts, torch.transpose(edges, 1, 2))
    else:
        print("Load error")

### Encoders
def get_geometry_encoder_args(contrastive_model_name):
    path_config = ContrastivePathConfig(contrastive_model_name)

    mesh_encoder_args_path = paths.HOME_PATH + path_config.get_model_args_path()

    # Load the encoder
    with open(mesh_encoder_args_path, "r") as f:
        model_args = json.load(f)

    return model_args

def instantiate_geometry_encoder(encoder_type, contrastive_model_name, device="cuda"):
    # Creates the model, but does not load the weights
    # Load the encoder
    model_args = get_geometry_encoder_args(contrastive_model_name)
    if encoder_type == "pcn" or encoder_type == "pc":
        encoder = DGCNN_param(args=model_args).to(device)
    elif encoder_type == "mesh_feast":
        encoder = FeaStNet(args=model_args).to(device)
    else:
        print("Error creating encoder")
        return
    return encoder


def create_cad_transformer(cad_checkpoint_name, device="cuda", phase="test"):
    cfg_cad = ConfigAE(exp_name=cad_checkpoint_name, phase=phase, overwrite=False, device=device)
    cad_load_path = paths.HOME_PATH + cfg_cad.get_checkpoint_path("latest")
    cad_encoder = VanillaCADTransformer(cfg_cad)

    cad_checkpoint = torch.load(cad_load_path, map_location='cpu', weights_only=True)
    cad_encoder.load_state_dict(cad_checkpoint['model_state_dict'])
    cad_encoder.eval()
    cad_encoder.to(device)

    return cad_encoder


def create_clip(encoder_type, contrastive_model_name, cad_encoder, encoder_ckpt_num=None, generation=True, device="cuda"):
    path_config = ContrastivePathConfig(contrastive_model_name)
    geometry_encoder = instantiate_geometry_encoder(encoder_type=encoder_type, contrastive_model_name=contrastive_model_name, device=device)

    checkpoint_path = path_config.get_checkpoint_path(checkpoint_num=encoder_ckpt_num)
    # if encoder_ckpt_num == "latest" or encoder_ckpt_num is None:
    #     checkpoint_path = paths.HOME_PATH + "results/Contrastive/" + contrastive_model_name + "model/trained_models/latest" + ".pth"
    # else:
    #     checkpoint_path = paths.HOME_PATH + "results/Contrastive/" + contrastive_model_name + "model/trained_models/backup/ckpt_epoch" + str(encoder_ckpt_num) + ".pth"

    clip_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True) # 'cpu'

    clip = CLIP(image_encoder=geometry_encoder, cad_encoder=cad_encoder, dim_latent=256, dim_image=256).to(device)
    clip.load_state_dict(clip_checkpoint['model_state_dict'])
    if generation:
        clip_adapter = DavinciClipAdapter(clip=clip)
    else:
        clip_adapter = XClipAdapter(clip=clip)
    clip.eval()
    return clip_adapter

def get_dataloader(encoder_type, phase, model_args, batch_size, data_root, use_normals, num_workers=24, mesh_folder="meshes/"):
    if encoder_type == "pcn" or encoder_type == "pc":
        return get_contrastive_cloud_dataloader(phase=phase, data_root=data_root, batch_size=batch_size, with_normals=use_normals, num_points=model_args["num_points"], num_workers=num_workers)
    elif encoder_type == "mesh_feast":
        # Enforce that batch size is 1
        return get_contrastive_mesh_feast_dataloader(phase=phase, data_root=data_root, batch_size=1, num_workers=num_workers, mesh_folder=mesh_folder, use_normals=use_normals)
    else:
        print("Load error")

class CADAutoencoder:
    def __init__(self, cad_checkpoint_name, device="cuda"):
        self.cad_autoencoder = create_cad_transformer(cad_checkpoint_name=cad_checkpoint_name, device=device)

    def cad_to_latent(self, command, args): # TODO check
        self.cad_autoencoder.forward(commands_enc=command, args_enc=args, z=None, return_tgt=False)

    def cad_latent_to_encoding(self, cad_latent):
        #input is [1, B, 256]
        outputs = self.cad_autoencoder.forward(commands_enc=None, args_enc=None, z=cad_latent, return_tgt=False)
        batch_out_vec = logits2vec(outputs)

        # append the first token because the model is autoregressive
        # Create a start token for each sequence in the batch
        begin_loop_vec = np.full((batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64) # Shape: (B, 1, 17) - batch_size × 1 position × feature_dim
        begin_loop_vec[:, :, 0] = SOL_IDX
        auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]  # (B, 60, 17)

        # out_vec = auto_batch_out_vec[0]
        return auto_batch_out_vec

class GeometryToCAD:
    def __init__(self, encoder_type, contrastive_model_name, encoder_ckpt_num, device="cuda"):
        self.geometry_encoder_args = get_geometry_encoder_args(contrastive_model_name)

        cad_checkpoint_name = self.geometry_encoder_args["cad_checkpoint_name"]
        self.cad_checkpoint_name = cad_checkpoint_name

        self.cad_encoder_wrapper = CADAutoencoder(cad_checkpoint_name, device=device)
        self.cad_encoder = self.cad_encoder_wrapper.cad_autoencoder

        self.generation_clip = create_clip(encoder_type, contrastive_model_name=contrastive_model_name, encoder_ckpt_num=encoder_ckpt_num,
                                           cad_encoder=self.cad_encoder, device=device, generation=True)
        self.retrieval_clip = create_clip(encoder_type, contrastive_model_name=contrastive_model_name, encoder_ckpt_num=encoder_ckpt_num,
                                          cad_encoder=self.cad_encoder, device=device, generation=False)

        self.use_normals = False
        if "use_normals" in self.geometry_encoder_args and self.geometry_encoder_args["use_normals"]:
            self.use_normals = True


    def encode_geometry(self, *args, normalize=False, generation=False):
        if generation:
            mesh_emb_tens = self.generation_clip.embed_image(*args, normalization=normalize)
        else:
            mesh_emb_tens, _ = self.retrieval_clip.embed_image(*args, normalization=normalize)
        return mesh_emb_tens

    def encode_cad(self, commands, args, normalize=False, generation=False):
        if generation:
            out = self.generation_clip.embed_cad((commands, args), normalization=normalize)
        else:
            out, _ = self.retrieval_clip.embed_cad((commands, args), normalization=normalize)
        return out

    def cad_latent_to_encoding(self, cad_latent):
        return self.cad_encoder_wrapper.cad_latent_to_encoding(cad_latent)


    def cad_latent_to_cad(self, cad_latent):
        outputs = self.cad_encoder.forward(commands_enc=None, args_enc=None, z=cad_latent, return_tgt=False)
        batch_out_vec = logits2vec(outputs)

        # append the first token because the model is autoregressive
        begin_loop_vec = np.full((batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64)
        begin_loop_vec[:, :, 0] = SOL_IDX
        auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]  # (B, 60, 17)

        out_vec = auto_batch_out_vec[0]

        # Cad vec to brep
        from cadlib.visualize import vec2CADsolid
        out_shape = vec2CADsolid(out_vec)
        return out_shape
 