# ----------------------------
# 
# Code borrowed from: https://github.com/ChrisWu1997/DeepCAD
#
#-----------------------------
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
# import random
from cadlib.macro import EOS_VEC
import numpy as np
import random
from multiprocessing import cpu_count

from cadlib.util import load_cad_vec

def cycle(dl):
    while True:
        for data in dl:
            yield data


def get_dataloader(phase, config, shuffle=None, num_workers=None, split_dataset=False):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    if num_workers is None:
        num_workers = cpu_count()

    dataset = CADDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=num_workers,
                            pin_memory = True)
    return dataloader


def create_dataloader(dataset, phase, batch_size, shuffle, num_workers, collate_fn=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle


    if num_workers is None:
        num_workers = cpu_count()
    num_workers = min(num_workers, cpu_count())

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_shuffle, num_workers=num_workers,
                            pin_memory = True, collate_fn=collate_fn,
                            drop_last=True)
    return dataloader


def get_single_cad_dataset(phase, config, data_ids=None, num_data=None, shuffle=None, num_workers=None, inflate_data_ratio=1):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    if num_workers is None:
        num_workers = cpu_count()

    dataset = AssortedCADDataset(phase, config, data_ids=data_ids, num_data=num_data, inflate_data_ratio=inflate_data_ratio)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=num_workers,
                            pin_memory = True)
    return dataloader


class AssortedCADDataset(Dataset):
    def __init__(self, phase, config, data_ids=None, num_data=None, inflate_data_ratio=1):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        assert not (num_data is None and data_ids is None) 
        if data_ids is not None:
            self.all_data = data_ids
        else:
            self.path = os.path.join(config.data_root, "filtered_data.json")
            self.all_data = []
            with open(self.path, "r") as fp:
                all_data = json.load(fp)[phase][:num_data]
            for data_id in tqdm(all_data, smoothing=0.1):
                h5_path = os.path.join(self.raw_data, str(data_id) + ".h5")
                try:
                    h5py.File(h5_path, "r")
                    self.all_data.append(data_id)
                except:
                    print("Issue in ", data_id)
                    continue
        self.num_data = len(self.all_data)
        self.inflate_data_ratio = inflate_data_ratio

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves        # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256

    def __getitem__(self, index):
        data_id = self.all_data[index % self.num_data]
        command, args = load_cad_vec(data_path=self.raw_data, data_id=data_id, pad=True, as_tensor=True, as_single_vec=False)
        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return self.num_data * self.inflate_data_ratio
        
class CADDataset(Dataset):
    def __init__(self, phase, config):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        if config.fine_tuning:
            self.path = os.path.join(config.data_root, "filtered_data_balanced.json")
        elif config.deepcad_splits:
            self.path = os.path.join(config.data_root, "filtered_data_deepcad.json")
        else:
            self.path = os.path.join(config.data_root, "filtered_data.json")
        self.all_data = []
        with open(self.path, "r") as fp:
            all_data = json.load(fp)[phase]

        # # check if data is valid:
        for data_id in tqdm(all_data, smoothing=0.1):
            h5_path = os.path.join(self.raw_data, str(data_id) + ".h5")
            try:
                h5py.File(h5_path, "r")
                self.all_data.append(data_id)
            except:
                print("Issue in ", data_id)
                continue

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        command, args = load_cad_vec(data_path=self.raw_data, data_id=data_id, pad=True, as_tensor=True, as_single_vec=False)

        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)





