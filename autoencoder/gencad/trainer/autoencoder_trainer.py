# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from cadlib.macro import (
        EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, 
        N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, 
        N_ARGS_EXT_PARAM, SOL_VEC
        )
import paths
from cadlib.util import compare_cad_vecs
from autoencoder.model_utils import logits2vec, sample_logits2vec
from cadlib.util import collate_cad_vec, separate_cad_vec
from utils.visual_logger import Logger

def plot_cad_vecs(out_vec, gt_vec, name, train_dir, phase, step):
        save_name = name[name.rfind("/"):name.rfind(".")]
        save_path = train_dir + f"/../{phase}_vis/epoch{step}/{save_name}.png"
        paths.mkdir(save_path)
        compare_cad_vecs(out_vec, gt_vec, out_file=save_path, names=("Out", "Target"))

class TrainerEncoderDecoder:
    def __init__(self, model, loss_fn, optimizer, config, scheduler=None):
        self.model = model  # CAD transformer 
        self.loss_fn = loss_fn  # CAD loss function
        self.optimizer = optimizer   
        self.lr = config.lr 
        self.device = config.device
        self.grad_clip = config.grad_clip
        self.val_every = config.val_every
        self.save_every = config.save_every
        self.num_epoch = config.num_epochs
        self.log_dir = config.log_dir
        self.model_path = config.model_dir

        self.scheduler = scheduler

        self.step, self.epoch = 0, 0

        # set up tensor board
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

        self.logger = Logger(save_dir=self.log_dir + "/", exp_name=config.exp_name)

    def _update_scheduler(self, epoch): 
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], epoch)
        self.scheduler.step()

    def _record_loss(self, loss_dict, mode="train"):
        # update loss in train or validation tensor board
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.step)


    def train_one_step(self, data): 
        # train for one step

        self.model.train()

        # print(data["id"])
        commands = data["command"].to(self.device)
        args = data["args"].to(self.device)

        outputs = self.model(commands, args)

        loss_dict = self.loss_fn(outputs)
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
       
        self.optimizer.step()

        return outputs, loss_dict


    def validate_one_step(self, data):
        self.model.eval() 
        commands = data['command'].to(self.device)
        args = data['args'].to(self.device)

        with torch.no_grad():
            outputs = self.model(commands, args)

        loss_dict = self.loss_fn(outputs)

        return outputs, loss_dict
    

    def eval_one_epoch(self, val_loader):
        self.model.eval()

        pbar = tqdm(val_loader)
        pbar.set_description("EVALUATE[{}]".format(self.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            commands = data['command'].to(self.device)
            args = data['args'].to(self.device)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            with torch.no_grad():
                logits = self.model(commands, args)
                # logits = {"command_logits": outputs['command_logits'], "args_logits": outputs['args_logits']}
                pred_vec = logits2vec(logits, refill_pad=False, to_numpy=True)

            # gt_vec = collate_cad_vec(gt_commands, gt_args)
            pred_cmd, pred_args = separate_cad_vec(pred_vec)
            # for i in range(10):
            #     plot_cad_vecs(pred_vec[i], gt_vec[i], name=data["id"][i], train_dir=self.model_path, phase="val", step=self.step)

            # out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
            # out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)


            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == pred_args).astype(int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])


        pred_vec[:, 1:] = pred_vec[:, 0:-1]
        pred_vec[:, 0] = SOL_VEC
        gt_vec = collate_cad_vec(data["command"], data["args"]).detach().cpu().numpy()
        for i in range(min(10, len(pred_vec))):
            plot_cad_vecs(pred_vec[i], gt_vec[i], name=data["id"][i], train_dir=self.model_path, 
                        phase="val", step=self.epoch-1)
                    
        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.epoch)

    def _load_ckpt(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise ValueError("Checkpoint {} not exists.".format(ckpt_path))
         
        checkpoint = torch.load(ckpt_path)
        print('# '* 25)
        print("Loading checkpoint from: {} ...".format(ckpt_path))

        # load model 
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print('--------- checkpoint loaded')
 

    def _save_ckpt(self, epoch=None):
        model_state_dict = self.model.state_dict()
        if epoch is None:
            save_path = os.path.join(self.model_path, "latest.pth")
        else:
            save_path = os.path.join(self.model_path, "backup", "ckpt_epoch{}.pth".format(epoch))
        paths.mkdir(save_path)
        torch.save({
            'model_state_dict': model_state_dict, 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()}, 
            save_path
        )
        save_path = os.path.join(self.model_path, "latest_epoch.log")
        with open(save_path, 'w') as f:
            f.write(str(self.epoch))

    def train(self, train_loader, val_loader, val_loader_all, ckpt=None):

        if ckpt is not None:
            self._load_ckpt(ckpt)

        for epoch in range(self.num_epoch):
            self.epoch += 1
            pbar = tqdm(train_loader)

            for b, data in enumerate(pbar):
                # train one epoch 
                outputs, loss_dict = self.train_one_step(data)
                
                # Save pic
                if epoch % self.save_every == 0 and b == 0:
                    pred_vec = logits2vec(outputs, refill_pad=False, to_numpy=True)
                    pred_vec[:, 1:] = pred_vec[:, 0:-1]
                    pred_vec[:, 0] = SOL_VEC
                    gt_vec = collate_cad_vec(data["command"], data["args"]).detach().cpu().numpy()
                    for i in range(min(len(pred_vec), 10)):
                        plot_cad_vecs(pred_vec[i], gt_vec[i], name=data["id"][i], train_dir=self.model_path, 
                                    phase="train", step=epoch)
                        
                # udpate tensorboard
                if self.step % 10 == 0:
                    self._record_loss(loss_dict, 'train')

                # update pbar
                pbar.set_description("EPOCH[{}][{}]".format(epoch, b))
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss_dict.items()}))

                self.step += 1

                # validate one step
                if self.step % self.val_every == 0: 
                    val_data = next(val_loader)
                    outputs, loss_dict  = self.validate_one_step(val_data)
                    self._record_loss(loss_dict, mode="validation")

                self._update_scheduler(epoch)
            


            # validation
            if self.epoch % 5 == 0: 
                self.eval_one_epoch(val_loader_all)

            # save model: checkpoint 
            self._save_ckpt()
            if self.epoch % self.save_every == 0: 
                pbar.set_description("saving model at: {}".format(self.model_path))
                self._save_ckpt(self.epoch)

        # save the final model 
        self._save_ckpt()
    
