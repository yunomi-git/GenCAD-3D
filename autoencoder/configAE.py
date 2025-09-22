import os
from datetime import datetime
import shutil
import sys 

from cadlib.macro import ARGS_DIM, N_ARGS, ALL_COMMANDS, MAX_N_EXT, \
                        MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN

import json
from .config_base import AutoencoderPathConfig

class ConfigAE(AutoencoderPathConfig):
    """
    Configuration of the autoencoder
    """
    def __init__(self, exp_name, load_from_json=False, override_kargs:dict=None,
                 phase="train", num_epochs=1,
                 lr=3e-4, batch_size=64,
                 load_ckpt=False,
                 save_every=5,
                 val_every=1,
                 device=None,
                 overwrite=True,
                 fine_tuning=False,
                 encoder_leave_n_unfrozen=1,
                 warmup_step=2000,
                 deepcad_splits=False,
                 use_group_emb=True,
                 dropout=0.2,
                 noise_tokens=False,
                 perturb_tokens=False,
                 data_root="data",
                 pos_encoding="LUT",
                 nonautoregressive=False,
                 use_scheduler=True,
                 nonauto_guidance=False,
                 weight_decay=0.0,
                 repredict=False,
                 layer_norm=True):
        super().__init__(exp_name=exp_name)
        
        # self.proj_dir = "results"
        # self.exp_name = exp_name
        # self.exp_train_name = exp_name + "/autoencoder"
        # self.exp_dir = os.path.join(self.proj_dir, self.exp_train_name)
        # self.model_dir = os.path.join(self.exp_dir, 'trained_models')        # results/experiment_name/log
        # self.log_dir = os.path.join(self.exp_dir, 'log')            # results/experiment_name/log

        self.phase = phase

        # --------- experiment parameters ------------
        self.load_ckpt = load_ckpt
        self.noise_tokens = noise_tokens
        self.perturb_tokens = perturb_tokens
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.val_every = val_every
        self.batch_size = batch_size
        self.weight_decay=weight_decay
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.warmup_step = warmup_step
        self.num_workers = 12   
        self.device = device
        self.overwrite = overwrite

        self.fine_tuning = fine_tuning
        self.encoder_leave_n_unfrozen = encoder_leave_n_unfrozen
        self.deepcad_splits = deepcad_splits

        self.pos_encoding = pos_encoding
        assert pos_encoding in ["LUT", "ROPE"]
        self.nonautoregressive = nonautoregressive
        self.layer_norm = layer_norm
        self.num_repredictions = 1
        self.repredict = repredict
        self.teacher_init_weight = 0.9
        self.teacher_decay_epochs = 200
        self.teacher_final_weight = 0.1

        self.nonauto_guidance = nonauto_guidance
        self.nar_init_weight = 0
        self.nar_decay_epochs = 600
        self.nar_final_weight = 0.4 

        # experiment paths
        # experiment directory: results/experiment_name
        self.data_root = data_root


        # ----------- model hyperparameters -----------
        self.n_enc_heads = 8                 # Transformer config: number of heads
        self.n_enc_layers = 4                # Number of Encoder blocks
        self.n_dec_heads = 8                 # Transformer config: number of heads
        self.n_dec_layers = 4                # Number of Encoder blocks        
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality
        self.dropout = dropout               # Dropout rate used in basic layers and Transformers
        self.dim_z = 256                 # Latent vector dimensionality
        self.use_group_emb = use_group_emb

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }
        self.max_num_groups = 30
        self.grad_clip = 1.0


        # ---------- CAD parameters: fixed ------------
        # ----- Config file is not updated with these information because they are fixed for all experiments

        self.args_dim = ARGS_DIM # 256
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)  # line, arc, circle, EOS, SOS

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES

        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN


        if load_from_json:
            # Load configuration from JSON file
            self.load_from_json(override_kargs)


        # =============== 
        # Post load actions
        # ===============
        if self.overwrite:
            if os.path.exists(self.exp_dir):
                response = input(f"Directory '{self.exp_dir}' already exists. Overwrite? (y/n): ").strip().lower()
                if response == 'y':
                    print("Overwriting the existing directory.")
                    shutil.rmtree(self.exp_dir)
                else: 
                    sys.exit("exit")  

            self.write_config()

        # create directories
        if phase=="train":
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)


        # self.is_train = phase == "train"

    @staticmethod
    def json_exists(exp_name):
        proj_dir = "results"
        exp_name = exp_name + "/autoencoder"
        exp_dir = os.path.join(proj_dir, exp_name)
        json_file = os.path.join(exp_dir, 'config.json')
        return os.path.exists(json_file)

    def save_as_json(self):
        json_file = os.path.join(self.exp_dir, 'config.json')
        data = vars(self).copy()
        for key, val in data.items():
            try:
                json.dumps(val)
            except:
                data[key] = None
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, override_kargs:dict=None):
        # assumes exp_dir is already set
        json_file = os.path.join(self.exp_dir, 'config.json')
        with open(json_file, "r") as f:
            data = json.load(f)

        print("=====")
        print("loading autoencoder with:")
        print(json.dumps(data, indent=4))
        
        for key, value in data.items():
            # if hasattr(self, key):
            setattr(self, key, value)
            # else:
            #     print(f"Warning: {key} not found in ConfigAE attributes.")
        
        if override_kargs is not None:
            for key, value in override_kargs.items():
                # if hasattr(self, key):
                setattr(self, key, value)
                # else:
                #     print(f"Warning: {key} not found in ConfigAE attributes.")


    def write_config(self):
        # save this configuration
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.phase == "train":
            config_path = os.path.join(self.exp_dir, f'config_autoencoder.txt')
            with open(config_path, 'w') as f:
                    f.write(f'Config for # # autoencoder model # # ' + '\n')
                    f.write(f'Time -> {current_time}' + '\n' + '\n')
                    f.write(f'  experiment directory -> {self.proj_dir}/{self.exp_train_name}/' + '\n')
                    f.write(f'  data directory -> {self.data_root}/' + '\n')
                    f.write(f'  load chekpoint: {self.load_ckpt}' + '\n' + '\n')
                    f.write(f'  num of workers: {self.num_workers}' + '\n')
                    f.write(f'  device: {self.device}' + '\n' + '\n')
                    f.write(f'  fine_tuning: {self.fine_tuning}' + '\n' + '\n')
                    f.write(f'  encoder_leave_n_unfrozen: {self.encoder_leave_n_unfrozen}' + '\n' + '\n')


                    f.write('\n' + '# '*25 + '\n')
                    f.write(f'  number of epochs: {self.num_epochs}' + '\n')
                    f.write(f'  batch size: {self.batch_size}' + '\n')
                    f.write(f'  learning rate: {self.lr}' + '\n')
                    f.write(f'  scheduler use: {self.use_scheduler}' + '\n')
                    f.write(f'        warmup steps: {self.warmup_step}' + '\n')
                    f.write(f'  dropout: {self.dropout}' + '\n')
                    f.write(f'  loss weights:' + '\n')
                    f.write(f'        command loss: {self.loss_weights["loss_cmd_weight"]}' + '\n')
                    f.write(f'        argument loss: {self.loss_weights["loss_args_weight"]}' + '\n')
                    
                    f.write(f'  save every: {self.save_every}' + '\n')
                    f.write(f'  validate every: {self.val_every}' + '\n' + '\n')
                    
                    f.write('\n' + '# '*25 + '\n')
                    f.write(f'  encoder -> heads: {self.n_enc_heads}, layers: {self.n_enc_layers}' + '\n')
                    f.write(f'  decoder -> heads: {self.n_dec_heads}, layers: {self.n_dec_layers}' + '\n')
                    f.write(f'  latent dimension: {self.dim_z}' + '\n')

                    f.write(f'  use group embedding: {self.use_group_emb}' + '\n')
                    f.write(f'        number of groups: {self.max_num_groups}' + '\n')

        