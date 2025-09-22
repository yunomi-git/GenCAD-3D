# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import torch
import torch_geometric.data
from lightning.fabric import Fabric
import os
from collections import OrderedDict
from tqdm import tqdm
from autoencoder.model_utils import AvgMeter
from contrastive.model import CLIP
from torch.utils.tensorboard import SummaryWriter
import paths
from utils.visual_logger import Logger
from contrastive.configContrastive import ConfigCCIP

class TrainerCCIPImprovedModel: 
    def __init__(self, model: CLIP, config: ConfigCCIP, optimizer, scheduler=None, fabric: Fabric=None, prof=None):
        self.model = model
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.batch_size = config.batch_size
        self.batch_accumulation = config.batch_accumulation
        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.device = config.device
        self.val_every = config.val_every
        self.log_dir = config.log_dir
        self.model_path = config.model_dir
        self.save_every = config.save_every
        self.prof = prof

        self.loss_meter = AvgMeter()
        self.epoch, self.step = 0, 0

        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))
        self.gradient_accumulation_steps = config.gradient_accumulation

        self.fabric = fabric
        self.distributed = False
        if self.fabric is not None:
            self.distributed = True

        if self.distributed:
            self.model, self.optimizer = fabric.setup(model, optimizer)
        else:
            self.model = model.to(config.device)

        self.last_state_dict = None
        self.last_optim_dict = None

        self.logger = Logger(save_dir=config.log_dir + "/", exp_name=config.exp_name[:config.exp_name.find("/")])

    def train_one_step(self, data_list, step):
        # train for one epoch
        self.model.train() 

        batch_cad_list = []
        batch_image_list = []
        for data in data_list:
            batch_cad, batch_image = self.data_to_batch(data)
            batch_cad_list.append(batch_cad)
            batch_image_list.append(batch_image)

        has_accumulated = step % self.gradient_accumulation_steps == 0


        if not self.distributed:
            loss = self.model(batch_cad_list, batch_image_list, return_loss=True, freeze_cad_encoder=True)
            loss /= self.gradient_accumulation_steps
            if torch.isnan(loss):
                print("Loss is Nan. Setting to 0 and skipping")
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            else:
                loss.backward()
        else:
            with self.fabric.no_backward_sync(self.model, enabled=not has_accumulated):
                loss = self.model(batch_cad_list, batch_image_list, return_loss=True, freeze_cad_encoder=True)
                loss /= self.gradient_accumulation_steps
                if torch.isnan(loss):
                    print("Loss is Nan. Setting to 0 and skipping")
                    loss = torch.tensor(0.0, requires_grad=True)
                else:
                    # loss.backward()
                    # needed since cannot set DDPStrategy(find_unused_parameters=True)
                    # fabric.backwards() does not interact well with unused parameters (frozen?)
                    for parameter in self.model.parameters():
                        loss += torch.sum(parameter) * 0
                    self.fabric.backward(loss)

        # Try to fix gradient. This did not work, This should not be an issue due to clip_norm though
        for param in self.model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print("NaN detected in gradients:", param)
                param.grad = torch.nan_to_num(param.grad, posinf=1.0, neginf=-1.0)

        if has_accumulated: # or (step + 1 == self.train_steps)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Try to save and fix parameters
            current_state_dict = self.model.state_dict()
            current_optim_dict = self.optimizer.state_dict()
            error_detected = False
            for param in current_state_dict.keys():
                if torch.isnan(current_state_dict[param]).any():
                    print("NAN detected in parameters. Resetting last parameters!", param)
                    current_state_dict[param] = self.last_state_dict[param]
                    error_detected = True
            for param in current_optim_dict['state'][0].keys():
                if torch.isnan(current_optim_dict['state'][0][param]).any():
                    print("NAN detected in optimizer. Resetting last parameters!", param)
                    current_optim_dict['state'][0][param] = self.last_state_dict['state'][0][param]
                    error_detected = True
            if error_detected:
                self.model.load_state_dict(current_state_dict)
                self.optimizer.load_state_dict(current_optim_dict)

            self.last_state_dict = self.model.state_dict().copy()
            self.last_optim_dict = self.optimizer.state_dict().copy()

            self.optimizer.zero_grad()

        return loss

    def data_to_batch(self, data):
        """
        Converts data from dataset to form to feed to model
        """
        # Convert cad to correct device
        if self.distributed:
            batch_cmd = self.fabric.to_device(data["command"])  # (B, 60)
            batch_args = self.fabric.to_device(data["args"])  # (B, 60, 16)
        else:
            batch_cmd = data["command"].to(self.device)  # (B, 60)
            batch_args = data["args"].to(self.device)  # (B, 60, 16)

        # Convert image to tuple
        if not isinstance(data["image"], tuple) and not isinstance(data["image"], list):
            data_image = (data["image"],)
        else:
            data_image = data["image"]

        # Convert to correct device
        batch_image = []
        for image in data_image:
            if isinstance(image, torch.Tensor) or isinstance(image, torch_geometric.data.Data):
                if self.distributed:
                    batch_image.append(self.fabric.to_device(image))
                else:
                    batch_image.append(image.to(self.device))
            else:
                batch_image.append(image)
        batch_image = tuple(batch_image)

        # contrastive loss
        batch_cad = (batch_cmd, batch_args)
        return batch_cad, batch_image

    def val_one_epoch(self, val_loader):
        self.model.eval()

        pbar = tqdm(range(len(val_loader) // self.batch_accumulation))
        pbar.set_description("EVALUATE[{}]".format(self.epoch))

        total_loss = 0.0

        iterate = iter(val_loader)
        for b in pbar:
            if self.prof is not None:
                self.prof.step()
            data_list = []
            for _ in range(self.batch_accumulation):
                data_list.append(next(iterate))

            batch_cad_list = []
            batch_image_list = []
            for data in data_list:
                batch_cad, batch_image = self.data_to_batch(data)

                batch_cad_list.append(batch_cad)
                batch_image_list.append(batch_image)

            loss = self.model(batch_cad_list, batch_image_list, return_loss=True, freeze_cad_encoder=True)

            pbar.set_postfix(OrderedDict({"val loss": loss.item()}))
            total_loss += loss.item()

        return total_loss

    def _save_ckpt(self, epoch=None, multi_gpu=False, only_image_encoder=False):
        if only_image_encoder:
            # save only the image encoder 
            if multi_gpu:
                model_state_dict = self.model.image_encoder.state_dict()
            else: 
                model_state_dict = self.model.module.image_encoder.state_dict()

            save_path = os.path.join(self.model_path, "img_encoder_ckpt_epoch{}.pth".format(self.epoch))
        else:
            # save the entire ccip model
            if multi_gpu:
                model_state_dict = self.model.module.state_dict()
            else: 
                model_state_dict = self.model.state_dict()

            if epoch is None:
                save_path = os.path.join(self.model_path, "latest.pth")
            else:
                save_path = os.path.join(self.model_path, "backup/ckpt_epoch{}.pth".format(self.epoch))

        paths.mkdir(save_path)


        if self.distributed:
            self.fabric.save(path=save_path,
                             state={
                                 'model_state_dict': model_state_dict,
                                 'optimizer_state_dict': self.optimizer.state_dict()
                             })
        else:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, save_path)

    def load_model(self, ckpt_path, only_image_encoder=False): 
        """
        load checkpoint for the model
        """
        if self.distributed:
            checkpoint = self.fabric.load(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path)

        if only_image_encoder:
            self.model.encode_image.load_state_dict(checkpoint['model_state_dict'])
            print('# # # # # Image encoder checkpoint loaded')
        else: 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('# # # # # CCIP model checkpoint loaded')


    def _record_loss(self, loss, mode="train"):
        # update loss in train or validation tensor board
        # losses_values = loss.item()  
        losses_values = loss      
        tb = self.train_tb if mode == 'train' else self.val_tb
        tb.add_scalar('training loss', losses_values, self.step)


    def train(self, train_loader, val_loader, distributed=False):
        if self.distributed:
            train_loader, val_loader = self.fabric.setup_dataloaders(train_loader, val_loader,
                                                                     move_to_device=False)

        for epoch in range(self.num_epochs):
            self.epoch += 1
            total_loss = 0.0

            # Why construct a list of batches? ans: for mesh models, must load/inference one meshes per batch
            iterate = iter(train_loader)
            pbar = tqdm(range(len(train_loader) // self.batch_accumulation))
            for b in pbar:
                if self.prof is not None:
                    self.prof.step()

                # Construct the list of batches
                data_list = []
                for _ in range(self.batch_accumulation):
                    data_list.append(next(iterate))

                train_loss = self.train_one_step(data_list, b)
                pbar.set_description("EPOCH[{}][{}]".format(epoch, b))
                pbar.set_postfix(OrderedDict({"train loss": train_loss.item()}))
                if self.step % 10 == 0:
                    self._record_loss(train_loss, mode='train')

                self.step += 1
                total_loss += train_loss.item()

            self.logger.add_to_log("losses", metric_name="train", value=total_loss, epoch=self.epoch)
            self.logger.plot_log("losses")
            if self.scheduler is not None:
                self.scheduler.step(total_loss)

            if self.epoch % self.val_every == 0:
                with torch.no_grad():
                    val_loss = self.val_one_epoch(val_loader)
                    pbar.set_description("validation")
                    pbar.set_postfix(OrderedDict({"validation loss": val_loss}))
                
                self._record_loss(val_loss, mode='validation')
                self.logger.add_to_log("losses", metric_name="val", value=val_loss, epoch=self.epoch)
                self.logger.plot_log("losses")
            if self.epoch % self.save_every == 0: 
                pbar.set_description("saving model at: {}".format(self.model_path))
                self._save_ckpt(multi_gpu=distributed, epoch=self.epoch)
            self._save_ckpt(multi_gpu=distributed)


        self._save_ckpt(multi_gpu=distributed)
