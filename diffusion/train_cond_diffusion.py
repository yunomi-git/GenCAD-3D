# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import h5py
import numpy as np
from pathlib import Path

from .diff_model.cond_gaussian_diffusion import GaussianDiffusion1D, Trainer1D, Dataset1D
from .diff_model.cond_models import ResNetDiffusion
from .generate_diffusion_embeddings import DiffusionEncodingConfig
import torch
import paths
import argparse

class DiffusionTrainConfig:
    def __init__(self, save_suffix, diffusion_encoding_config: DiffusionEncodingConfig):
        self.save_suffix = save_suffix
        self.diffusion_encoding_config = diffusion_encoding_config
        self.training_save_dir = diffusion_encoding_config.diffusion_save_dir + save_suffix

# Run with
# but first generate embeddings
# python -m diffusion.train_cond_diffusion -num_workers 24 -encoder_type pc -results_path pcn_b64_BalNAFT_OrigData -checkpoint 300
# Set:
# - encoder_type
# - results_path
# - ckpt
# before this, run generate_diffusion_embeddings
if __name__=="__main__":
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Train different models.")
    parser.add_argument("-encoder_type", "--encoder_type", type=str, required=True, help="pc, pcn, mesh_feast")
    parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=True, help="name of contrastive model")
    parser.add_argument("-checkpoint", "--checkpoint", type=str, required=True, help="contrastive checkpoint")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("-dataset", "--dataset", type=str, required=False, default="GenCAD3D", help="name of the dataset to encode")
    args = parser.parse_args()

    encoder_type = args.encoder_type
    contrastive_model_name = args.contrastive_model_name
    checkpoint = args.checkpoint
    dataset = args.dataset

    encoder_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, checkpoint, dataset=dataset)

    diffusion_batch_size = 2048
    save_suffix = "batch2048/"
    diffusion_config = DiffusionTrainConfig(save_suffix, encoder_config)

    model = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1, dropout_second=0.1, d_out=256)

    diffusion = GaussianDiffusion1D(
        model,
        z_dim=256,
        timesteps = 500,
        objective = 'pred_x0',
        auto_normalize=False
    )

    with h5py.File(encoder_config.cad_embeddings_path, 'r') as f:
        cad_latent_data = f["train_zs"][:]

    with h5py.File(encoder_config.geometry_embeddings_path, 'r') as f:
        geometry_latent_data = f["train_zs"][:]

    # remove nans
    valid_indices = np.unique(np.where(~np.isnan(geometry_latent_data))[0])
    cad_latent_data = cad_latent_data[valid_indices]
    geometry_latent_data = geometry_latent_data[valid_indices]

    cad_tensor = torch.tensor(cad_latent_data)
    geometry_tensor = torch.tensor(geometry_latent_data)

    dataset = Dataset1D(cad_tensor, geometry_tensor)

    batch_size = diffusion_batch_size

    save_folder_name = paths.HOME_PATH + diffusion_config.training_save_dir
    Path(save_folder_name).mkdir(parents=True, exist_ok=True)
    trainer = Trainer1D(
            diffusion,
            device=torch.device("cuda"),
            dataset = dataset,
            train_batch_size = batch_size,
            train_lr = 1e-5,
            train_num_steps = 1000000,         # total training steps
            gradient_accumulate_every = 4096//batch_size,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,
            results_folder=save_folder_name,
            save_and_sample_every=125000,
            gt_data_path=encoder_config.cad_embeddings_path,
            fabric=None,
            num_workers=args.num_workers
        )

    trainer.train()
# 