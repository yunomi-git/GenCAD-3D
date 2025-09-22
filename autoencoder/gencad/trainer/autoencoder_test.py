import sys
import os
from collections import OrderedDict
import h5py
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from cadlib.macro import (
        EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, 
        N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, 
        N_ARGS_EXT_PARAM, EOS_IDX, MAX_TOTAL_LEN
        )
from autoencoder.model_utils import logits2vec
from autoencoder.cad_dataset import get_dataloader
from utils.util import context_print
from autoencoder.configAE import ConfigAE

class TestEncoderDecoder:
    def __init__(self, model, ckpt_path, config: ConfigAE, save_subfolder):
        self.model = model  # CAD transformer 
        self.ckpt_path = ckpt_path
        self.device = config.device
        self.exp_name = config.exp_name
        self.batch_size = config.batch_size
        self.subfolder=save_subfolder

        # load checkpoint
        self._load_ckpt()


    def _load_ckpt(self): 
        if not os.path.exists(self.ckpt_path):
            raise ValueError("Checkpoint {} not exists.".format(self.ckpt_path))
         
        checkpoint = torch.load(self.ckpt_path)
        print('# '* 25)
        print("Loading checkpoint from: {} ...".format(self.ckpt_path))

        # load model 
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print('--------- checkpoint loaded')

    def reconstruct(self, config, test_loader):
        
        # create output directory
        # output_dir = "results/" + self.exp_name + "/reconstructions"
        output_dir = config.exp_dir + "/eval/" + self.subfolder + "/reconstructions"
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval() 
        pbar = tqdm(test_loader)
        print("Total number of test data:", len(test_loader))

        for b, data in enumerate(pbar):
            batch_size = data['command'].shape[0]
            commands = data['command'].to(self.device)
            args = data['args'].to(self.device)

            # ground truth vector
            gt_vec = torch.cat([commands.unsqueeze(-1), args], dim=-1).squeeze(1).detach().cpu().numpy()
            commands_ = gt_vec[:, :, 0]

            with torch.no_grad():
                outputs = self.model(commands, args)
            
            batch_out_vec = logits2vec(outputs)

            # append the first token because the model is autoregressive
            begin_loop_vec = gt_vec[:, 0, :]  # (B, 17)
            begin_loop_vec = begin_loop_vec[:, np.newaxis, :]  # (B, 1, 17)
            auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]  # (B, 60, 17)

            for j in range(batch_size):
                try:
                    out_vec = auto_batch_out_vec[j]
                    seq_len = commands_[j].tolist().index(EOS_IDX)

                    data_id = data["id"][j].split('/')[-1]

                    save_path = os.path.join(output_dir, '{}.h5'.format(data_id))
                    with h5py.File(save_path, 'w') as fp:
                        fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=int)
                        fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=int)
                except:
                    context_print("EOS not found")

    def encode(self, config): 
        # create output directory
        output_dir = "results/" + self.exp_name + "/encoder_output"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'encoded_data.h5')
        fp = h5py.File(save_path, 'w')

        self.model.eval() 

        for phase in ['train', 'test', 'validation']:
            print(f'--------- Encoding {phase} data')

            data_loader = get_dataloader(phase=phase, config=config)

            pbar = tqdm(data_loader)
            latent_data = []


            for b, data in enumerate(pbar):
                batch_size = data['command'].shape[0]
                commands = data['command'].to(self.device)
                args = data['args'].to(self.device)

                # ground truth vector
                gt_vec = torch.cat([commands.unsqueeze(-1), args], dim=-1).squeeze(1).detach().cpu().numpy()
                commands_ = gt_vec[:, :, 0]


                with torch.no_grad():
                    latent_outputs = self.model(commands, args, encode_mode=True)
                    latent_outputs = latent_outputs.squeeze()  # (B, 256)
                
                latent_data.append(latent_outputs.detach().cpu().numpy())

            latent_data = np.concatenate(latent_data, axis=0)
            # save latent dataset 
            fp.create_dataset('{}_zs'.format(phase), data=latent_data)

            print(f'--------- data saved at: {output_dir}')
            print('# '*20)


    def decode(self, config, z_path=None, z=None, save=True): 

        # create output directory
        output_dir = "results/" + self.exp_name + "/decoder_output/" + z_path.split('.')[0].split('/')[-1] + '_dec'
        os.makedirs(output_dir, exist_ok=True)


        # load latent data ---> (N, 256)
        if z_path is not None:
            with h5py.File(z_path, 'r') as fp:
                z_keys = list(fp.keys())
                zs = fp[z_keys[0]][:]

        # or directly give the latent vector
        if z is not None:
            zs = z 

        self.model.eval() 

        # decode
        for i in range(0, len(zs), config.batch_size):
            with torch.no_grad():
                batch_z = torch.tensor(zs[i:i+config.batch_size], dtype=torch.float32).unsqueeze(1)
                batch_z = batch_z.cuda()
                outputs = self.model(None, None, z=batch_z, return_tgt=False)
                batch_out_vec = logits2vec(outputs)
                # begin loop vec: [4, -1, -1, ...., -1] 
                begin_loop_vec = np.full((batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64)
                begin_loop_vec[:, :, 0] = 4

                auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]  # (B, 60, 17)

            for j in range(len(batch_z)):
                out_vec = auto_batch_out_vec[j]
                out_command = out_vec[:, 0]
                try:
                    seq_len = out_command.tolist().index(EOS_IDX)

                    save_path = os.path.join(output_dir, '{}.h5'.format(i + j))
                    if save:
                        with h5py.File(save_path, 'w') as fp:
                            fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=int)
                except:
                    print("cannot find EOS")

        print(f'--------- data saved at: {output_dir}')
