import os
import glob
import numpy as np
import h5py
# from joblib import Parallel, delayed
import multiprocessing as mp
import argparse
import sys
sys.path.append("..")
from utils import write_ply
from cadlib.visualize import vec2CADsolid, CADsolid2pc


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
args = parser.parse_args()



SAVE_DIR = args.src + '_pc'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def process_one(path):
    data_id = path.split("/")[-1]
    data_id = data_id[:8]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    # print("[processing] {}".format(data_id))
    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(float)

    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        print(e)  # FIXME: Inspect the error message
        return None

    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None
    
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)


all_paths = glob.glob(os.path.join(args.src, "*.h5"))

# Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in all_paths)
mp.Pool(8).map(process_one, all_paths)

