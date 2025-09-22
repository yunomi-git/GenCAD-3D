# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import json
import torch
import torch.optim as optim
import argparse
import faulthandler

from third_party.FeaStNet.models import FeaStNet
from contrastive.dgcnn_model import DGCNN_param
from contrastive.configContrastive import ConfigCCIP, ContrastivePathConfig
from autoencoder.configAE import ConfigAE
from contrastive.datasets import get_contrastive_cloud_dataloader, get_contrastive_mesh_feast_dataloader
import paths
from contrastive.trainer import (TrainerCCIPImprovedModel)
from autoencoder.gencad.model import VanillaCADTransformer
from contrastive.model import CLIP


def load_dgcnn(args, config, use_normals):
    if args.pretrained_contrastive_model_name is not None:
        pretrained_config = ContrastivePathConfig(args.pretrained_contrastive_model_name)
        mesh_encoder_args_path = paths.HOME_PATH + pretrained_config.get_model_args_path()

        # Get encoder args
        with open(mesh_encoder_args_path, "r") as f:
            model_args = json.load(f)
        use_normals = model_args["use_normals"]

    else:
        input_dims = 3
        if use_normals:
            input_dims = 6

        print("USE_NORMALS IS SET TO", use_normals)

        model_args = {
            "num_points": 2048,
            "use_normals": use_normals,
            "input_dims": input_dims,
            "conv_channel_sizes": [128, 128, 256, 512],
            "emb_dims": 512,
            "linear_sizes": [1024, 512],
            "num_outputs": 256,
            "k": 20,
            "dropout": 0.2,
            "normalize": False
        }

    cloud_encoder = DGCNN_param(model_args)

    # data loader: each batch contains: command, args, clouds and id
    train_loader = get_contrastive_cloud_dataloader(phase="train", data_root=config.data_root, batch_size=config.batch_size,
                                             with_normals=use_normals, num_points=model_args["num_points"], num_workers=args.num_workers,
                                             normalize=model_args["normalize"])
    val_loader = get_contrastive_cloud_dataloader(phase="validation", data_root=config.data_root, batch_size=config.batch_size,
                                           with_normals=use_normals, num_points=model_args["num_points"], num_workers=args.num_workers,
                                           normalize=model_args["normalize"])

    return cloud_encoder, model_args, train_loader, val_loader


def load_feastnet(args, config, use_normals):
    if args.pretrained_contrastive_model_name is not None:
        pretrained_config = ContrastivePathConfig(args.pretrained_contrastive_model_name)
        mesh_encoder_args_path = paths.HOME_PATH + pretrained_config.get_model_args_path()

        # Get encoder args
        with open(mesh_encoder_args_path, "r") as f:
            model_args = json.load(f)
        use_normals = model_args["use_normals"]

    else:
        in_channels = 6 if use_normals else 3
        model_args = {
            "in_channels": in_channels,
            "num_outputs": 256,
            "heads": 10,
            "t_inv": True,
            "outputs_at": "global",
            "conv_dims": [128, 128, 256, 256],  # default 16, 32, 64
            "lin_dims": [512, 256],  # default 128 258
            "use_normals": use_normals,
        }

    cloud_encoder = FeaStNet(model_args)

    print(model_args)
    mesh_folder =  "meshes/"
    if args.mesh_directory is not None:
        mesh_folder = args.mesh_directory
    # data loader: each batch contains: command, args, images and id
    train_loader = get_contrastive_mesh_feast_dataloader(phase="train", data_root=config.data_root, batch_size=config.batch_size, num_workers=args.num_workers, mesh_folder=mesh_folder, use_normals=use_normals)
    val_loader = get_contrastive_mesh_feast_dataloader(phase="validation", data_root=config.data_root, batch_size=config.batch_size, num_workers=args.num_workers, mesh_folder=mesh_folder, use_normals=use_normals)

    return cloud_encoder, model_args, train_loader, val_loader


### ======================================================================================
###
### ======================================================================================
def train_encoder(args, fabric=None, prof=None):
    distributed = fabric is not None
    if fabric is None:
        total_devices = 1
    else:
        total_devices = fabric.world_size

    autoencoder_model_name = args.autoencoder_model_name

    # autoencoder_config = AutoencoderPathConfig(autoencoder_model_name)
    cfg_cad = ConfigAE(exp_name=autoencoder_model_name, phase=phase, overwrite=False) # This is a legacy config. will need to update
    cad_load_path = cfg_cad.get_checkpoint_path("latest")
    cad_encoder = VanillaCADTransformer(cfg_cad)

    if distributed:
        cad_checkpoint = fabric.load(cad_load_path)
    else:
        cad_checkpoint = torch.load(cad_load_path, map_location='cpu', weights_only=True)
    cad_encoder.load_state_dict(cad_checkpoint['model_state_dict'])
    print('\n # # # # # CAD encoder checkpoint loaded: ' + str(autoencoder_model_name))

    # Optimization params
    num_epochs = 300
    lr = 1e-3

    clip_batch_size = args.batch_size
    load_batch_size = min(64, clip_batch_size)
    if "mesh" in args.encoder_type:
        load_batch_size = 1
    gradient_batch_size = 128 // total_devices

    gradient_accumulation = gradient_batch_size // clip_batch_size
    batch_accumulation = clip_batch_size // load_batch_size

    if args.dataset is None:
        dataset = "GenCAD3D"
    else:
        dataset = args.dataset
        print("USING DATASET", dataset)

    data_root = paths.DATA_PATH +  dataset + "/"

    config = ConfigCCIP(contrastive_model_name=args.exp_name,
                        phase=phase, num_epochs=num_epochs,
                        lr=lr, batch_size=load_batch_size,
                        encoder_type=args.encoder_type,
                        save_every=args.save_every,
                        device=device,
                        data_root=data_root,
                        gradient_accumulation=gradient_accumulation,
                        cad_checkpoint_name=args.autoencoder_model_name,
                        overwrite=False,
                        gradient_batch_size = gradient_batch_size * total_devices,
                        batch_accumulation=batch_accumulation)

    # Load mesh encoder
    if args.encoder_type == "pc":
        encoder, model_args, train_loader, val_loader = load_dgcnn(args, config, use_normals=args.use_normals)
    elif args.encoder_type == "mesh_feast":
        encoder, model_args, train_loader, val_loader = load_feastnet(args, config, use_normals=args.use_normals)
    else:
        print("encoder type not implemented:", args.encoder_type)
        return

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Number of parameters: {total_params}")

    model_args["num_parameters"] = total_params
    model_args["autoencoder_model_name"] = args.autoencoder_model_name
    model_args["dataset"] = dataset

    model_dir = config.model_dir
    # save the mesh encoder params
    with open(model_dir + "/../" + "model_args.json", "w") as f:
        json.dump(model_args, f)

    # model
    clip = CLIP(image_encoder=encoder, cad_encoder=cad_encoder, dim_latent=256, dim_image=256, update_cad_grad=True)
    # use_saved_checkpoint = False
    if args.pretrained_contrastive_model_name is not None:
        pretrained_contrastive_config = ContrastivePathConfig(args.pretrained_contrastive_name)
        checkpoint_path = paths.HOME_PATH + pretrained_contrastive_config.get_checkpoint_path(args.pretrained_contrastive_checkpoint)
        saved_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # encoder_checkpoint = extract_checkpoint(full_checkpoint=saved_checkpoint, model_name="image_encoder", append_name_prefix=False)
        # optim_checkpoint = saved_checkpoint["optimizer_state_dict"]
        # scheduler_checkpoint = saved_checkpoint["scheduler_state_dict"]
        print("USING PRETRAINED", checkpoint_path)
        print("Loading pretrained encoder")
        clip.load_state_dict(saved_checkpoint)
        # print("Loading pretrained optimizer")
        # optimizer.load_state_dict(optim_checkpoint)
        # optimizer_to(optimizer, device)
        # scheduler.load_state_dict(scheduler_checkpoint)

    # keep cad encoder frozen
    for param in clip.cad_encoder.parameters():
        param.requires_grad = False

    # optimizer
    optimizer = optim.AdamW(clip.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.8)

    ccip_trainer = TrainerCCIPImprovedModel(model=clip, config=config, optimizer=optimizer, scheduler=scheduler, fabric=fabric, prof=prof)
    ccip_trainer.train(train_loader=train_loader, val_loader=val_loader)

if __name__ == "__main__":
    torch.manual_seed(100)
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Train different models.")

    parser.add_argument("-encoder_type", "--encoder_type", type=str, required=True, help="pc, pcn, mesh")
    parser.add_argument("-name", "--exp_name", type=str, required=True, help="experiment numbers, keep fixed for the same experiment")
    parser.add_argument("-autoencoder_model_name", "--autoencoder_model_name", type=str, default=None)#, choices={"Autoencoder_SynthBal_1MFT", "Autoencoder_SynthBal_FT"}
    parser.add_argument("-dataset", "--dataset", type=str, default=None)
    parser.add_argument("-use_normals", "--use_normals", action="store_true", help="Append normals to dataset")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)

    parser.add_argument("-gpu", "--gpu", type=int, default=0, help="gpu device number, multi-gpu not supported")
    parser.add_argument("-pretrain_name", "--pretrained_contrastive_model_name", type=str, default=None, help="name of pretrained contrastive model")
    parser.add_argument("-pretrain_ckpt", "--pretrained_contrastive_checkpoint", type=str, default=None, help="pretrained checkpoint to train on")

    parser.add_argument("-num_workers", "--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("-save_every", "--save_every", type=int, default=10)
    parser.add_argument("-mesh_directory", "--mesh_directory", type=str, default=None)

    args = parser.parse_args()

    print("num workers", args.num_workers)

    print(args)

    device = torch.device(f"cuda")
    phase = "train"
    faulthandler.enable()

    train_encoder(args)