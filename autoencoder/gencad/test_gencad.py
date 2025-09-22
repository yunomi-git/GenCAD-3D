# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import torch
import argparse
from autoencoder.cad_dataset import get_dataloader
from ..configAE import ConfigAE
from .model import VanillaCADTransformer
from autoencoder.gencad.trainer import TestEncoderDecoder
from cadlib.util import compare_cad_vecs
import os
import h5py
from tqdm import tqdm
import paths


def plot_comparison(result_dir):
    reconstruction_dir = result_dir + "reconstructions/"
    filenames = sorted(os.listdir(reconstruction_dir))[:30]

    for name in tqdm(filenames):
        path = os.path.join(reconstruction_dir, name)
        with h5py.File(path, "r") as fp:
            out_vec = fp["out_vec"][:].astype(int)
            gt_vec = fp["gt_vec"][:].astype(int)

        save_name = name[:name.rfind(".")]
        save_path = result_dir + f"vis/{save_name}.png"
        paths.mkdir(save_path)
        compare_cad_vecs(out_vec, gt_vec, out_file=save_path, names=("Out", "Target"))
        print(out_vec[:, 0])
        print(gt_vec[:, 0])
        # plt.show()

def test_model(args=None):
    # set phase to testing for configuration 
    phase = "test" 
    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}")
    data_root = paths.DATA_PATH + "GenCAD3D/"

    # This is a legacy config. need to update
    config = ConfigAE(exp_name=args.exp_name, 
            phase=phase, batch_size=args.batch_size, 
            device=device, data_root=data_root,
            deepcad_splits=True, # This tests on the same splits as the original deepcad
            overwrite=False)

    # test data loader
    test_loader = get_dataloader(phase=phase, config=config, num_workers=16)

    # model 
    model = VanillaCADTransformer(config).to(config.device) 
    checkpoint_path = config.get_checkpoint_path(args.ckpt_name)

    test_agent = TestEncoderDecoder(model=model, ckpt_path=checkpoint_path, config=config, save_subfolder=args.subfolder)

    test_agent.reconstruct(config, test_loader=test_loader)

    if args.eval:
        # visualize a few
        plot_comparison(result_dir=config.exp_dir + "/eval/" + args.subfolder + "/")


if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="Train different models.")

    parser.add_argument("-name", "--exp_name", type=str, required=True, help="name of the experiment")
    parser.add_argument("-subfolder", "--subfolder", type=str, required=True)

    parser.add_argument("-b", "--batch_size", type=int, default=512, required=False, help="batch size")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, required=False, help="device number: 0, 1, 2, 3")
    parser.add_argument("-ckpt", "--ckpt_name", type=str, default="latest", required=False, help="latest, or 100, 200, etc..")
    parser.add_argument("-eval", "--eval", action='store_true', default=False, help="evaluate model and save data")

    args = parser.parse_args()
    test_model(args)

