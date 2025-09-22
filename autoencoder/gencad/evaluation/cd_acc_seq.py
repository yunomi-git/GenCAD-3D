import os
import glob
import h5py
import numpy as np
import argparse
from joblib import Parallel, delayed
import random
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
import time
import sys
sys.path.append("..")
from utils import read_ply
from cadlib.visualize import vec2CADsolid, CADsolid2pc

import matplotlib.pyplot as plt 


# Load the sequence lengths
seq_lens = np.load('proj_log/baseline/results/seq_lens.npy')

# Calculate initial bins using np.histogram to get 10 bins
counts, initial_bins = np.histogram(seq_lens, bins=5)

# Define the bin edges as integers, rounding the initial bin edges
bin_edges = np.linspace(np.min(seq_lens), np.max(seq_lens), 5)
bin_edges = np.ceil(bin_edges).astype(int)

seq_len_cd_dict =  {bin_edges[k]: {'dist': []} for k in range(len(bin_edges))}
seq_len_cd_list  = list(seq_len_cd_dict.keys())



PC_ROOT = "data/pc_cad"
# data that is unable to process
SKIP_DATA = [""]
n_points = 2000


def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points


def process_one(path):
    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(float)
        # gt_vec = fp["gt_vec"][:].astype(np.float)

    data_id = path.split('/')[-1].split('.')[0][:8]
    truck_id = data_id[:4]
    gt_pc_path = os.path.join(PC_ROOT, truck_id, data_id + '.ply')

    if not os.path.exists(gt_pc_path):
        return None

    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return None
    
    try:
        out_pc = CADsolid2pc(shape, n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None

    if np.max(np.abs(out_pc)) > 2: # normalize out-of-bound data
        out_pc = normalize_pc(out_pc)

    gt_pc = read_ply(gt_pc_path)
    sample_idx = random.sample(list(range(gt_pc.shape[0])), n_points)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    return cd


src = "proj_log/baseline/results/test_1000"
filepaths = sorted(glob.glob(os.path.join(src, "*.h5")))
filepaths = filepaths[:-1]

save_path = src + '_pc_stat.txt'
record_res = None
if os.path.exists(save_path):
    response = input(save_path + ' already exists, overwrite? (y/n) ')
    if response == 'y':
        os.system("rm {}".format(save_path))
        record_res = None
    else:
        with open(save_path, 'r') as fp:
            record_res = fp.readlines()
            n_processed = len(record_res) - 3


dists = []
for i in tqdm(range(len(filepaths))):
    # print("processing[{}] {}".format(i, filepaths[i]))
    data_id = filepaths[i].split('/')[-1].split('.')[0]

    if record_res is not None and i < n_processed:
        record_dist = record_res[i].split('\t')[-1][:-1]
        record_dist = None if record_dist == 'None' else eval(record_dist)
        dists.append(record_dist)
        continue

    if data_id in SKIP_DATA:
        # print("skip {}".format(data_id))
        res = None
    else:
        res = process_one(filepaths[i])
    with open(save_path, 'a') as fp:
        print("{}\t{}\t{}".format(i, data_id, res), file=fp)

    with h5py.File(filepaths[i], 'r') as fp:
        out_vec = fp["out_vec"][:].astype(float)
        gt_vec = fp["gt_vec"][:].astype(float)

    gt_cmd = gt_vec[:, 0]

    if res is not None:
        if gt_cmd.shape[0] <= list(seq_len_cd_dict.keys())[0]:
            seq_len_cd_dict[seq_len_cd_list[0]]["dist"].append(res)
            
        elif gt_cmd.shape[0] <= list(seq_len_cd_dict.keys())[1] and gt_cmd.shape[0] > list(seq_len_cd_dict.keys())[0]:
            seq_len_cd_dict[seq_len_cd_list[1]]["dist"].append(res)
    
        elif gt_cmd.shape[0] <= list(seq_len_cd_dict.keys())[2] and gt_cmd.shape[0] > list(seq_len_cd_dict.keys())[1]:
            seq_len_cd_dict[seq_len_cd_list[2]]["dist"].append(res)
    
        elif gt_cmd.shape[0] <= list(seq_len_cd_dict.keys())[3] and gt_cmd.shape[0] > list(seq_len_cd_dict.keys())[2]:
            seq_len_cd_dict[seq_len_cd_list[3]]["dist"].append(res)
    
        elif gt_cmd.shape[0] <= list(seq_len_cd_dict.keys())[4] and gt_cmd.shape[0] > list(seq_len_cd_dict.keys())[3]:
            seq_len_cd_dict[seq_len_cd_list[4]]["dist"].append(res)

np.save('proj_log/baseline/results/seq_len_cd_dict_deepcad.npy', seq_len_cd_dict)