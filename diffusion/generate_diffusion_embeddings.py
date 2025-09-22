import paths
import json
import contrastive.contrastive_evaluation_util as mesh_evaluation_util
from contrastive.configContrastive import ContrastivePathConfig
from contrastive.model import freeze_model_and_make_eval_
import os
import h5py
from tqdm import tqdm
import torch
import numpy as np
import torch_geometric
from contrastive.contrastive_evaluation_util import get_geometry_encoder_args

np.seterr(over='raise')

class DiffusionConfig:
    def __init__(self, encoder_type, contrastive_model_name, checkpoint=None, normalize=False, dataset=None):
        self.encoder_type = encoder_type
        self.checkpoint = checkpoint
        self.normalize = normalize

        contrastive_config = ContrastivePathConfig(contrastive_model_name)
        mesh_encoder_args_path = paths.HOME_PATH + contrastive_config.get_model_args_path()
        # Load the encoder
        with open(mesh_encoder_args_path, "r") as f:
            self.clip_json = json.load(f)

        self.cad_checkpoint_name = self.clip_json["cad_checkpoint_name"]

        if checkpoint == "latest" or checkpoint is None:
            checkpoint_name = "latest"
        else:
            checkpoint_name = "ckpt-" + str(checkpoint)

        self.diffusion_save_dir = "results/Diffusion/" + contrastive_model_name + "/" + checkpoint_name

        if normalize:
            self.diffusion_save_dir += "_normalize"

        self.diffusion_save_dir += "/"

        if dataset is not None:
            self.diffusion_save_dir += dataset + "/"


class DiffusionEncodingConfig(DiffusionConfig):
    def __init__(self, encoder_type, contrastive_model_name, checkpoint, normalize=False, dataset=None):
        super().__init__(encoder_type, contrastive_model_name, checkpoint, normalize, dataset)

        self.embedding_save_dir = self.diffusion_save_dir + "encoder_output/"

        # self.cad_checkpoint_name = cad_checkpoint_name
        self.geometry_embeddings_path = os.path.join(self.embedding_save_dir, 'geometry_embeddings.h5')
        self.cad_embeddings_path = os.path.join(self.embedding_save_dir, 'cad_embeddings.h5')

        geometry_encoder_args = get_geometry_encoder_args(contrastive_model_name)
        self.use_normals = False
        if "use_normals" in geometry_encoder_args and geometry_encoder_args["use_normals"]:
            self.use_normals = True


def data_to_batch(data):
    """
    Converts data from dataset to form to feed to model
    """
    # Convert cad to correct device
    batch_cmd = data["command"].to(device)  # (B, 60)
    batch_args = data["args"].to(device)  # (B, 60, 16)

    # Convert image to tuple
    if not isinstance(data["image"], tuple) and not isinstance(data["image"], list):
        data_image = (data["image"],)
    else:
        data_image = data["image"]

    # Convert to correct device
    batch_image = []
    for image in data_image:
        if isinstance(image, torch.Tensor) or isinstance(image, torch_geometric.data.Data):
            batch_image.append(image.to(device))
        else:
            batch_image.append(image)
    batch_image = tuple(batch_image)

    # contrastive loss
    batch_cad = (batch_cmd, batch_args)
    return batch_cad, batch_image

def encode(clip, cad_encoder, encoder_config: DiffusionEncodingConfig, dataloaders, normalize=False):
    # create output directory
    output_dir = encoder_config.embedding_save_dir
    os.makedirs(output_dir, exist_ok=True)
    fp_cloud = h5py.File(encoder_config.geometry_embeddings_path, 'w')
    fp_cad = h5py.File(encoder_config.cad_embeddings_path, 'w')

    freeze_model_and_make_eval_(clip)
    freeze_model_and_make_eval_(cad_encoder)

    for phase in ['train', 'test', 'validation']:
        print(f'--------- Encoding {phase} data')

        data_loader = dataloaders[phase]

        pbar = tqdm(data_loader)
        cad_latent_data = []
        cloud_latent_data = []

        for b, data in tqdm(enumerate(pbar)):

            batch_cad, batch_image = data_to_batch(data)

            with torch.no_grad():
                try:
                    cloud_latent_outputs = clip.embed_image(*batch_image, normalization=normalize)
                    cad_latent_outputs = clip.embed_cad(batch_cad, normalization=normalize)
                    # This is the same as the original
                    # cad_latent_outputs = cad_encoder(batch_cad[0], batch_cad[1], encode_mode=True)
                except Exception as e:
                    print("encoding error, skipping:", e)
                    continue
                # cad_latent_outputs = cad_latent_outputs.squeeze()  # (B, 256)
                if len(cad_latent_outputs.shape) == 1:
                    cad_latent_outputs = cad_latent_outputs.unsqueeze(0)
                if len(cloud_latent_outputs.shape) == 1:
                    cloud_latent_outputs = cloud_latent_outputs.unsqueeze(0)

            cloud_latent_data.append(cloud_latent_outputs.detach().cpu().numpy())
            cad_latent_data.append(cad_latent_outputs.detach().cpu().numpy())

        cloud_latent_data = np.concatenate(cloud_latent_data, axis=0)
        cad_latent_data = np.concatenate(cad_latent_data, axis=0)

        print(cloud_latent_data.shape)
        print(cad_latent_data.shape)

        # save latent dataset
        fp_cloud.create_dataset('{}_zs'.format(phase), data=cloud_latent_data)
        fp_cad.create_dataset('{}_zs'.format(phase), data=cad_latent_data)

        print(f'--------- data saved at: {output_dir}')
        print('# ' * 20)


# Run with
# python -m diffusion.generate_diffusion_embeddings -encoder_type pc -results_path pcn_b64_BalNAFT_OrigData -checkpoint 300 -mesh_folder meshes_rm2kfix -remote h100 -cad_checkpoint_name Balanced_NA_FT_unf
# set:
# - encoder_type
# - results_path
# - checkpoint
# - data_root
# - mesh_folder (if meshes)
# - cad_encoder

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train different models.")

    parser.add_argument("-encoder_type", "--encoder_type", type=str, required=True, help="pc, pcn, mesh_feast")
    parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=True, help="path starting from content root")
    parser.add_argument("-checkpoint", "--checkpoint", type=str, required=True, help="contrastive model checkpoint")
    parser.add_argument("-mesh_folder", "--mesh_folder", type=str, required=False, default="meshes_rm2kfix", help="path starting from content root")
    parser.add_argument("-dataset", "--dataset", type=str, required=False, default="GenCAD3D", help="name of the dataset to encode")
    args = parser.parse_args()

    device = "cuda"
    encoder_type = args.encoder_type
    contrastive_model_name = args.contrastive_model_name + "/"
    checkpoint = args.checkpoint
    mesh_folder = args.mesh_folder + "/"

    print("Using checkpoint", contrastive_model_name, checkpoint)

    dataset = args.dataset

    encoder_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, checkpoint, dataset=dataset)

    batch_size = 32
    print("Using dataset", dataset)
    data_root = paths.DATA_PATH + dataset + "/"

    output_dir = encoder_config.embedding_save_dir

    cad_encoder = mesh_evaluation_util.create_cad_transformer(cad_checkpoint_name=encoder_config.cad_checkpoint_name, device=device)
    clip = mesh_evaluation_util.create_clip(contrastive_model_name=contrastive_model_name, encoder_ckpt_num=checkpoint,
                                            cad_encoder=cad_encoder, device=device, generation=True, encoder_type=encoder_type)

    encoder_args = mesh_evaluation_util.get_geometry_encoder_args(contrastive_model_name)
    dataloaders = {}
    for phase in ["train", "test", "validation"]:
        dataloaders[phase] = mesh_evaluation_util.get_dataloader(encoder_type=encoder_type, phase=phase,
                                                                 model_args=encoder_args, data_root=data_root,
                                                                 batch_size=batch_size,
                                                                 use_normals=encoder_args["use_normals"],
                                                                 mesh_folder=mesh_folder)

    encode(clip=clip, cad_encoder=cad_encoder, encoder_config=encoder_config, dataloaders=dataloaders, normalize=False)


