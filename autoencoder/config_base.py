import os
import json

class AutoencoderPathConfig:
    def __init__(self, exp_name):
        self.proj_dir = "results/Autoencoder/"
        self.exp_name = exp_name
        self.exp_train_name = exp_name + "/autoencoder"
        self.exp_dir = os.path.join(self.proj_dir, self.exp_train_name)
        self.model_dir = os.path.join(self.exp_dir, 'trained_models')        # results/experiment_name/log
        self.log_dir = os.path.join(self.exp_dir, 'log')            # results/experiment_name/log

    def get_checkpoint_path(self, checkpoint=None):
        if checkpoint is None or checkpoint == "latest":
            checkpoint_path = self.model_dir + "/latest.pth"
        else:
            checkpoint_path = self.model_dir + "/backup/ckpt_epoch" + str(checkpoint) + ".pth"
        return checkpoint_path


    @staticmethod
    def json_exists(exp_name):
        proj_dir = "results/Autoencoder/"
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

        # print("=====")
        # print("loading autoencoder with:")
        # print(json.dumps(data, indent=4))
        
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