import h5py
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import sys, os
import json
import multiprocessing as mp
sys.path.append("..")
from cadlib.visualize import vec2CADsolid, CADsolid2pc
from cadlib.macro import *
from joblib import Parallel, delayed
from glob import glob
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--analyze_brep', action='store_true')
parser.add_argument('--compare_shape', action='store_true')
parser.add_argument('--unique', action='store_true')
args = parser.parse_args()

result_dir = args.src

filenames = sorted(os.listdir(result_dir))


def is_valid(path) -> bool:
    """
    Returns:
        valid_sample: bool
        valid_shapes: TopoDS_Shape | None
        data_id: int
        out_vec: np.array | None
    """
    with h5py.File(path, 'r') as fp:
        out_vec = fp['out_vec'][:].astype(int)

    data_id = path.split('/')[-1].split('.')[0]
    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return False, None, data_id, None
    
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print('convert pc failed', data_id)
        return False, None, data_id, None
    
    if args.analyze_brep:
        analyzer = BRepCheck_Analyzer(shape)
        if not analyzer.IsValid():
            print("validity check failed", data_id)
            return False, None, data_id, None
    
    return True, shape, data_id, out_vec  # valid(bool), valid_shape, data_id, valid_vector


def is_unique(sample, idx) -> bool:
    for i, target in enumerate(valid_queries):
        if idx == i:
            continue
        if args.compare_shape and sample.IsPartner(target):
            return False
        if isinstance(sample, np.ndarray) and np.array_equal(sample, target):
            return False
    return True

unique_samples = []

if args.parallel:
    valid_samples, valid_shapes, data_ids, valid_vectors = zip(*mp.Pool(processes=8).map(is_valid, tqdm([os.path.join(result_dir, name) for name in filenames])))
    # valid_samples, valid_shapes, data_ids, valid_vectors = zip(*Parallel(n_jobs=8, verbose=2)(delayed(is_valid)(os.path.join(result_dir, name)) for name in filenames))

    # print(type(valid_shapes), valid_shapes[0], len(valid_shapes))
    valid_shapes = list(filter(None, valid_shapes))  # Eliminate None i.e. invalid shapes
    # print(type(valid_vectors), type(valid_vectors[0]), len(valid_vectors))
    valid_vectors = list(filter(lambda array: array is not None, valid_vectors))
    valid_queries = valid_shapes if args.compare_shape else valid_vectors
    assert len(valid_shapes) == len(valid_vectors), 'this line can be removed later'

    invalid_data_ids = np.array(data_ids)[np.array(valid_samples) == False]

    if args.unique:
        unique_samples = mp.Pool(8).starmap(is_unique, tqdm([(item, i) for i, item in enumerate(valid_queries)]))
        # unique_samples = Parallel(n_jobs=8, verbose=2, prefer='processes')(delayed(is_unique)(item, i) for i, item in enumerate(valid_queries))
        
else:
    raise NotImplementedError
    valid_samples = []
    for name in tqdm(filenames):
        path = os.path.join(result_dir, name)
        valid_samples.append(is_valid(path))

validity = np.array(valid_samples).mean()
uniqueness = np.array(unique_samples).mean()
print(len(valid_shapes))
print(f'Validity: {validity}, Uniqueness: {uniqueness}')
save_path = result_dir + '_3metrics.txt'
with open(save_path, 'w') as f:
    log_str = f'<Validity>\nN_TOTAL={len(valid_samples)}, N_VALID={len(valid_shapes)}, Validity={validity}\n\n' + \
              f'<Uniqueness>\nN_TOTAL={len(unique_samples)}, N_UNIQUE={np.array(unique_samples).sum()}, Uniqueness={uniqueness}\n'
    log_str += f'\nAnalyze_brep: {args.analyze_brep}\nCompare_shape: {args.compare_shape}\n'
    log_str += '\nInvalid data ids:\n'
    for data_id in invalid_data_ids:
        log_str += f'{data_id}\n'
    f.write(log_str)

