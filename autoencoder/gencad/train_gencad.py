# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import torch
import torch.optim as optim
import argparse

from autoencoder.config_base import AutoencoderPathConfig
from ..configAE import ConfigAE
from .loss import CADLoss
from ..cad_dataset import get_dataloader
from utils.scheduler import GradualWarmupScheduler
import paths

from .trainer import TrainerEncoderDecoder
from .model import VanillaCADTransformer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train_model(args=None):

    device = torch.device(f"cuda:{args.gpu}")
    phase = "train" 

    # Transformer autoencoder
    if args.data_root is None:
        data_root = paths.DATA_PATH + "GenCAD3D/"
    else:
        data_root = args.data_root

    warmup_step = args.warm_up
    # if args.fine_tune:
    #     warmup_step = 200

    config = ConfigAE(exp_name=args.exp_name,
            phase=phase, num_epochs=args.num_epochs,
            lr=args.lr, batch_size=args.batch_size, use_group_emb=False,
            save_every=args.save_every, data_root=data_root, fine_tuning=args.fine_tune,
            warmup_step=warmup_step,
            device=device, overwrite=False)
    config.save_as_json()

    # data loader: each batch contains: command, args and id
    train_loader = get_dataloader(phase="train", config=config, num_workers=args.num_workers)
    val_loader_all = get_dataloader(phase="validation", config=config, num_workers=args.num_workers)

    val_loader = get_dataloader(phase="validation", config=config, num_workers=args.num_workers)
    val_loader = cycle(val_loader)

    
    # model 
    model = VanillaCADTransformer(config).to(device)
    if config.fine_tuning and config.encoder_leave_n_unfrozen > -1:
        model.freeze_encoder_to_final_nth_layer(config.encoder_leave_n_unfrozen)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # 1 unfrozen = 4829718
    # 0 frozen =   6715478

    # loss function
    loss_fn = CADLoss(config)
    # optimizer
    optimizer = optim.Adam(model.parameters(), config.lr)

    if config.use_scheduler: 
        scheduler = GradualWarmupScheduler(optimizer, 1.0, config.warmup_step)
    else: 
        scheduler = None

    ae_trainer = TrainerEncoderDecoder(model, loss_fn, optimizer, config, scheduler)

    pretrain_path = None
    if args.pretrained_name is not None:
        pretrain_config = AutoencoderPathConfig(args.pretrained_name)
        pretrain_path = pretrain_config.get_checkpoint_path(args.pretrain_checkpoint)

    # train model 
    ae_trainer.train(train_loader=train_loader, val_loader=val_loader, val_loader_all=val_loader_all, ckpt=pretrain_path)



if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="Train different models.")
    # parser.add_argument("model", choices=["autoencoder", "ccip", "ccip_improved", "ldm", "ldm_cond", "diffusion_prior", "decoder"], help="Model to train")
    torch.manual_seed(100)

    # if "autoencoder" in sys.argv:
        # command to run: 
        # python train_gencad.py  -name test1 -epoch 1000 -lr 1e-3 -b 512 -gpu 0 -sf 200
        # For fine tuning:
        # python -m train_gencad  -name autoencoder_balanced_NA_finetune -epoch 100 -lr 1e-4 -b 512 -gpu 0 -sf 50 --fine_tune --encoder_leave_n_unfrozen -1 -ckpt results/autoencoder_balanced_NA/autoencoder/trained_models/ckpt_epoch1000.pth
    parser.add_argument("-name", "--exp_name", type=str, required=True, help="experiment numbers, keep fixed for the same experiment")
    parser.add_argument("-epoch", "--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, help="gpu device number, multi-gpu not supported")
    parser.add_argument("-sf", "--save_every", type=int, required=True, help="save every model every nth iteration")
    parser.add_argument("-nw", "--num_workers", type=int, default=16, required=False, help="num workers")
    parser.add_argument("-pretrain_name", "--pretrained_name", default=None, type=str, required=False, help="load from checkpoint")
    parser.add_argument("-pretrain_ckpt", "--pretrained_checkpoint", default=None, type=str, required=False, help="load from checkpoint")

    parser.add_argument("-wu", "--warm_up", default=2000, type=int, required=False, help="warmup steps")
    parser.add_argument("-data_root", "--data_root", default=None, type=str, required=False, help="source of data")
    parser.add_argument("-fine_tune", "--fine_tune", action='store_true', default=False, help="set up fine-tuning with autoencoder freezing")
    # parser.add_argument("-freeze_n", "--encoder_leave_n_unfrozen", type=int, default=1, required=False, help="if fine-tuning, freezes encoder layers leaving n final layers unfrozen. set to -1 to avoid freezing anything")
    parser.add_argument("-train_assorted", "--train_assorted", type=int, default=None)

    args = parser.parse_args()
    train_model(args)


