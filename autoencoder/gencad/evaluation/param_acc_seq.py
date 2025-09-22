import h5py
from tqdm import tqdm
import os
import argparse
import numpy as np
import sys
sys.path.append("..")
from cadlib.macro import *
import bisect

import matplotlib.pyplot as plt 

# Load the sequence lengths
seq_lens = np.load('proj_log/baseline/results/seq_lens.npy')

# Calculate initial bins using np.histogram to get 10 bins
counts, initial_bins = np.histogram(seq_lens, bins=5)

# Define the bin edges as integers, rounding the initial bin edges
bin_edges = np.linspace(np.min(seq_lens), np.max(seq_lens), 5)
bin_edges = np.ceil(bin_edges).astype(int)

seq_len_dict =  {bin_edges[k]: {'param_acc': [], 'cmd_acc': []} for k in range(len(bin_edges))}
seq_len_list  = list(seq_len_dict.keys())

TOLERANCE = 3

result_dir = "proj_log/baseline/reconstructions/"
filenames = sorted(os.listdir(result_dir))

# overall accuracy
avg_cmd_acc = [] # ACC_cmd
avg_param_acc = [] # ACC_param

# accuracy w.r.t. each command type
each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

# accuracy w.r.t each parameter
args_mask = CMD_ARGS_MASK.astype(float)
N_ARGS = args_mask.shape[1]
each_param_cnt = np.zeros([*args_mask.shape])
each_param_acc = np.zeros([*args_mask.shape])

for name in tqdm(filenames):
    path = os.path.join(result_dir, name)
    with h5py.File(path, "r") as fp:
        out_vec = fp["out_vec"][:].astype(int)
        gt_vec = fp["gt_vec"][:].astype(int)

    out_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    out_param = out_vec[:, 1:]
    gt_param = gt_vec[:, 1:]

    cmd_acc = (out_cmd == gt_cmd).astype(int)
    param_acc = []
    for j in range(len(gt_cmd)):
        cmd = gt_cmd[j]
        each_cmd_cnt[cmd] += 1
        each_cmd_acc[cmd] += cmd_acc[j]
        if cmd in [SOL_IDX, EOS_IDX]:
            continue

        if out_cmd[j] == gt_cmd[j]: # NOTE: only account param acc for correct cmd
            tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(int)
            # filter param that do not need tolerance (i.e. requires strictly equal)
            if cmd == EXT_IDX:
                tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(int)[-2:]
            elif cmd == ARC_IDX:
                tole_acc[3] = (out_param[j] == gt_param[j]).astype(int)[3]

            valid_param_acc = tole_acc[args_mask[cmd].astype(bool)].tolist()
            param_acc.extend(valid_param_acc)

            each_param_cnt[cmd, np.arange(N_ARGS)] += 1
            each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc

    param_acc = np.mean(param_acc)
    avg_param_acc.append(param_acc)
    cmd_acc = np.mean(cmd_acc)
    avg_cmd_acc.append(cmd_acc)

    if gt_cmd.shape[0] <= list(seq_len_dict.keys())[0]:
        seq_len_dict[seq_len_list[0]]["param_acc"].append(param_acc)
        seq_len_dict[seq_len_list[0]]["cmd_acc"].append(cmd_acc)
        
    elif gt_cmd.shape[0] <= list(seq_len_dict.keys())[1] and gt_cmd.shape[0] > list(seq_len_dict.keys())[0]:
        seq_len_dict[seq_len_list[1]]["param_acc"].append(param_acc)
        seq_len_dict[seq_len_list[1]]["cmd_acc"].append(cmd_acc)
 
    elif gt_cmd.shape[0] <= list(seq_len_dict.keys())[2] and gt_cmd.shape[0] > list(seq_len_dict.keys())[1]:
        seq_len_dict[seq_len_list[2]]["param_acc"].append(param_acc)
        seq_len_dict[seq_len_list[2]]["cmd_acc"].append(cmd_acc)
 
    elif gt_cmd.shape[0] <= list(seq_len_dict.keys())[3] and gt_cmd.shape[0] > list(seq_len_dict.keys())[2]:
        seq_len_dict[seq_len_list[3]]["param_acc"].append(param_acc)
        seq_len_dict[seq_len_list[3]]["cmd_acc"].append(cmd_acc)
 
    elif gt_cmd.shape[0] <= list(seq_len_dict.keys())[4] and gt_cmd.shape[0] > list(seq_len_dict.keys())[3]:
        seq_len_dict[seq_len_list[4]]["param_acc"].append(param_acc)
        seq_len_dict[seq_len_list[4]]["cmd_acc"].append(cmd_acc)

np.save('proj_log/baseline/results/seq_len_dict_davinci.npy', seq_len_dict)