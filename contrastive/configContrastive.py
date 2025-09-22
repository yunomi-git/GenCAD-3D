import os
from datetime import datetime
import shutil
import sys 
import paths

class ContrastivePathConfig:
    def __init__(self, config_model_name):
        self.contrastive_model_name = config_model_name
        self.save_dir = "results/Contrastive/"
        self.results_path = self.save_dir + self.contrastive_model_name + "/"
        self.mesh_encoder_args_path = self.results_path + "model/model_args.json"

        self.exp_name = self.contrastive_model_name + "/model"
        self.exp_dir = os.path.join(self.save_dir, self.exp_name)
        self.model_dir = os.path.join(self.exp_dir, 'trained_models')        # results/experiment_name/log
        self.log_dir = os.path.join(self.exp_dir, 'log')            # results/experiment_name/log

    def get_checkpoint_path(self, checkpoint_num=None):
        if checkpoint_num == "latest" or checkpoint_num is None:
            checkpoint_path = paths.HOME_PATH + self.results_path + "model/trained_models/latest" + ".pth"
        else:
            checkpoint_path = paths.HOME_PATH + self.results_path + "model/trained_models/backup/ckpt_epoch" + str(
                checkpoint_num) + ".pth"

        return checkpoint_path

    def get_model_args_path(self):
        return self.mesh_encoder_args_path

class ConfigCCIP(ContrastivePathConfig):
    """
    Configuration of the autoencoder
    """
    def __init__(self, encoder_type,
                 contrastive_model_name="test_model",
                 phase="train", num_epochs=1,
                 lr=3e-4, batch_size=64,
                 load_ckpt=False,
                 save_every=1,
                 val_every=10,
                 device=None,

                 gradient_accumulation=1,
                 data_root="data/",
                 batch_accumulation=1,
                 gradient_batch_size=128,
                 cad_checkpoint_name="DeepCAD",
                 overwrite=False):

        super().__init__(contrastive_model_name)
        
        self.is_train = phase == "train"

        # --------- experiment parameters ------------
        self.data_root = data_root

        self.load_ckpt = load_ckpt
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.val_every = val_every
        self.batch_size = batch_size
        self.batch_accumulation = batch_accumulation
        self.gradient_batch_size = gradient_batch_size
        self.lr = lr  
        self.num_workers = 12   
        self.device = device
        self.encoder_type = encoder_type
        self.overwrite = overwrite
        self.gradient_accumulation = gradient_accumulation

        self.cad_checkpoint_name = cad_checkpoint_name

        # check if directories exit
        if self.overwrite:
            if os.path.exists(self.exp_dir):
                response = input(f"Directory '{self.exp_dir}' already exists. Overwrite? (y/n): ").strip().lower()
                if response == 'y':
                    print("Overwriting the existing directory.")
                    shutil.rmtree(self.exp_dir)
                else: 
                    sys.exit("exit") 

        # create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # ----------- model hyperparameters -----------

        self.write_config()

    def write_config(self):
        # save this configuration
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.is_train:
            config_path = os.path.join(self.exp_dir, f'config_clip_model.txt')
            with open(config_path, 'w') as f:
                    f.write(f'Config for # # clip model # # ' + '\n')
                    f.write(f'Time -> {current_time}' + '\n' + '\n')
                    f.write(f'  experiment directory -> {self.save_dir}/{self.exp_name}/' + '\n')
                    f.write(f'  data directory -> {self.data_root}' + '\n')
                    f.write(f'  cad_checkpoint_name: {self.cad_checkpoint_name}' + '\n' + '\n')

                    f.write('# '*25 + '\n' + '\n')
                    f.write(f'  number of epochs: {self.num_epochs}' + '\n')
                    # f.write(f'  load_batch size: {self.batch_size}' + '\n')
                    f.write(f'  clip_batch size: {self.batch_size * self.batch_accumulation}' + '\n')
                    f.write(f'  gradient batch size: {self.gradient_batch_size}' + '\n')
                    f.write(f'  learning rate: {self.lr}' + '\n')
                    f.write(f'  device: {self.device}' + '\n' + '\n')

                    f.write(f'  encoder_type: {self.encoder_type}' + '\n' + '\n')
                    f.write(f'  save every: {self.save_every}' + '\n')
                    f.write(f'  validate every: {self.val_every}' + '\n' + '\n')
                    f.write(f'  gradient_accumulation: {self.gradient_accumulation}' + '\n' + '\n')
                    f.write(f'  batch_accumulation: {self.batch_accumulation}' + '\n' + '\n')

                    # f.write(f'  gradient_accumulation: {self.gradient_accumulation}' + '\n' + '\n')
                    # f.write(f'  gradient_accumulation: {self.gradient_accumulation}' + '\n' + '\n')

                    f.write('# '*25 + '\n' + '\n')
