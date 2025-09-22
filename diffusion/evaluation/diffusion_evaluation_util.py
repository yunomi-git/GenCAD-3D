# This generates meshes
from trimesh.path.packing import paths

import torch
import numpy as np
import h5py
from ..diff_model.cond_gaussian_diffusion import GaussianDiffusion1D, Trainer1D, Dataset1D
from ..diff_model.cond_models import ResNetDiffusion
import paths
from ..generate_diffusion_embeddings import DiffusionEncodingConfig
from ..train_cond_diffusion import DiffusionTrainConfig
from contrastive.contrastive_evaluation_util import CADAutoencoder

class DiffusionSamplingConfig(DiffusionTrainConfig):
    def __init__(self, save_suffix, encoder_config: DiffusionEncodingConfig, diffusion_checkpoint):
        super().__init__(save_suffix, diffusion_encoding_config=encoder_config)
        self.diffusion_checkpoint = diffusion_checkpoint
        self.visualization_dir = ""
        self.eval_samples_output_dir = self.training_save_dir + "evaluation_samples/" + "ckpt" + str(diffusion_checkpoint) + "/"
        self.eval_reconstruction_output_dir = self.training_save_dir + "evaluation_reconstructions/"
        self.eval_iou_output_dir = self.training_save_dir + "evaluation_iou/"

        if diffusion_checkpoint == "latest" or diffusion_checkpoint is None:
            ckpt_name = "latest"
        else:
            ckpt_name = "model-" + str(self.diffusion_checkpoint)
        self.diffusion_checkpoint_path = self.training_save_dir + "models/" + ckpt_name + ".pt"


class DiffusionSampler:
    def __init__(self, embedding_config: DiffusionEncodingConfig, diffusion_sampling_config: DiffusionSamplingConfig, device="cuda"):
        ## create diffusion prior
        model = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1, dropout_second=0.1, d_out=256)

        self.diffusion = GaussianDiffusion1D(
            model,
            z_dim=256,
            timesteps = 500,
            objective = 'pred_x0',
            auto_normalize=False
        )

        self.diffusion.to(device)
        checkpoint_path = paths.HOME_PATH + diffusion_sampling_config.diffusion_checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.diffusion.load_state_dict(checkpoint['model'])
        self.diffusion.eval()

        with h5py.File(paths.HOME_PATH + embedding_config.geometry_embeddings_path, 'r') as f:
            geometry_latent_data = f["test_zs"][:]

        # remove nans
        orig_num = len(geometry_latent_data)
        valid_indices = np.unique(np.where(~np.isnan(geometry_latent_data))[0])
        # cad_latent_data = cad_latent_data[valid_indices]
        geometry_latent_data = geometry_latent_data[valid_indices]
        print("nan values in dataset:", orig_num - len(geometry_latent_data))

        geometry_tensor = torch.tensor(geometry_latent_data)

        self.geometry_latents = geometry_tensor.to(device)

        # CAD decoder for validation
        self.CAD_decoder = CADAutoencoder(cad_checkpoint_name=embedding_config.cad_checkpoint_name, device=device)


    def sample(self, geometry_embedding):
        cad_embedding = self.diffusion.sample(cond=geometry_embedding).unsqueeze(1)
        return cad_embedding

    def sample_single_force_valid(self, geometry_embedding, self_intersection_check=False, num_retries=16):
        # geometry embedding should be [B, dim] where B=1
        # This can only operate on 1 embedding at a time

        from cadlib.visualize import vec2CADsolid, vec2CADsolid_valid_check

        batch_geometry_embeddings = geometry_embedding.repeat(num_retries, 1)
        batch_cad_embedding = self.diffusion.sample(cond=batch_geometry_embeddings).unsqueeze(1)

        for cad_embedding in batch_cad_embedding:
            cad_encoding = self.CAD_decoder.cad_latent_to_encoding(cad_embedding.unsqueeze(0))[0]
            if not self_intersection_check:
                try:
                    vec2CADsolid(cad_encoding)
                    # If successful, break out
                    return cad_embedding
                except:
                    continue
            else:
                result = vec2CADsolid_valid_check(cad_encoding)
                if result is None:
                    continue
                else:
                    return cad_embedding

        return None

    def sample_N_force_valid(self, geometry_embedding, num_samples, self_intersection_check=False, num_retries=3):
        # geometry embedding should be [B, dim] where B=1
        # This can only operate on 1 embedding at a time

        from cadlib.visualize import vec2CADsolid, vec2CADsolid_valid_check

        batch_geometry_embeddings = geometry_embedding.repeat(num_samples * num_retries, 1)
        batch_cad_embedding = self.diffusion.sample(cond=batch_geometry_embeddings).unsqueeze(1)
        samples = []
        for cad_embedding in batch_cad_embedding:
            cad_encoding = self.CAD_decoder.cad_latent_to_encoding(cad_embedding.unsqueeze(0))[0]
            if not self_intersection_check:
                try:
                    vec2CADsolid(cad_encoding)
                    # If successful, break out
                    samples.append(cad_embedding)
                except:
                    continue
            else:
                result = vec2CADsolid_valid_check(cad_encoding)
                if result is None:
                    continue
                else:
                    samples.append(cad_embedding)

            if len(samples) == num_samples:
                break
        return samples
        # return None
