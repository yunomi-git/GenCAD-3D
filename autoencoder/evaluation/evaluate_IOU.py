from paths import DirectoryPathManager
from utils.util import DictList
from third_party.SolidAlign import align_shapes
from cadlib.visualize import vec2CADsolid_valid_check
from OCC.Extend.DataExchange import write_step_file
import numpy as np
from tqdm import tqdm
import cadquery as cq
from pathlib import Path
import paths
from geometry.geometry_data import GeometryLoader
import argparse
from utils.multi_proc_queue import MultiProcQueueProcessing, ERROR_STATUS
import pickle
import os
import h5py
import contextlib

def get_iou(target_file, source_file):
    try:
        target = cq.importers.importStep(target_file)
    except Exception as e:
        # print(e, target_file)
        raise Exception("Cannot Load")
    try:
        source = cq.importers.importStep(source_file)
    except Exception as e:
        print(e, source_file)
        raise Exception("Cannot Load")
    try:
        aligned_source, nIOU = align_shapes(source, target)
    except Exception as e:
        raise Exception("Error " + target_file, e)
    return nIOU


def generate_step_files_from_encodings(encodings_path, save_directory):
    print("Generating STEP files from reconstructed encodings")
    # These already exist from the reconstruction tests
    directory_manager = DirectoryPathManager(encodings_path, base_unit_is_file=True)
    filenames = directory_manager.get_file_names(extension=False)

    for filename in tqdm(filenames, smoothing=0.1):
        # load the cad vec
        path = os.path.join(encodings_path, filename) + ".h5"
        with h5py.File(path, "r") as fp:
            out_vec = fp["out_vec"][:].astype(int)

            gen_shape = vec2CADsolid_valid_check(out_vec)
            if gen_shape is None:
                continue
            save_path = save_directory + f'{filename[:8]}.step'
            with contextlib.redirect_stdout(None):
                write_step_file(gen_shape, save_path)

parser = argparse.ArgumentParser(description="Full Eval Pipeline")
parser.add_argument("-src", "--reconstruction_source", type=str, required=False, default=None)
parser.add_argument("-nw", "--num_workers", type=int, required=False, default=20)
parser.add_argument("-gen_step", "--gen_step", action="store_true")

args = parser.parse_args()

#7137 gencad
if __name__=="__main__":
    num_workers = args.num_workers

    encodings_path = paths.HOME_PATH + args.reconstruction_source
    gen_step = args.gen_step

    # This describes where to find the target Step files to compare to
    data_root = paths.DATA_PATH + "GenCAD3D/"
    geometry_loader = GeometryLoader(data_root=data_root, phase="test")

    #######################################
    # Generate Step Files
    save_parent_directory = "/".join(encodings_path.split("/")[:-2]) + "/"
    samples_dir = save_parent_directory + "evaluation_iou/"
    Path(samples_dir).mkdir(parents=True, exist_ok=True)
    print("reconstructions path", encodings_path)
    print("step outputs", samples_dir)
    if gen_step:
        generate_step_files_from_encodings(encodings_path, samples_dir)

    #######################################
    # IOU calculation
    multi_process = MultiProcQueueProcessing(args_global=None, task=None, num_workers=num_workers)
    folder_manager = DirectoryPathManager(samples_dir, base_unit_is_file=True)
    all_files = folder_manager.get_file_names(extension=False)

    process_list = all_files

    def task(name, step_dir, samples_dir):
        data_id = name[:4] + "/" + name
        target_file = step_dir + data_id + ".step"
        gen_file = samples_dir + name + ".step"
        # load cad vec to get the SL
        seq_len = geometry_loader.calc_cad_sequence_length(data_id)
        try:
            iou = get_iou(target_file, gen_file)
        except Exception as e:
            print(e)
            return ERROR_STATUS
        if iou > 1.001:
            print("iou>1 detected. error", iou)
            return ERROR_STATUS
        return seq_len, iou

    processor = MultiProcQueueProcessing(args_global=(geometry_loader.step_dir, samples_dir), task=task, num_workers=22)
    outputs = processor.process(process_list, timeout=1000)
    sl_dict = DictList()
    for seq_len, iou in outputs:
        sl_dict.add_to_key(key=seq_len, value=iou)

    iou_by_sl = sl_dict.dictionary

    dic_save_path = save_parent_directory + "eval_recon_iou_by_seq_len.pkl"
    dic_to_save = {
        'avg_iou': iou_by_sl,
    }
    with open(dic_save_path, 'wb') as f:
        pickle.dump(dic_to_save, f)

    # print stats
    print(encodings_path)

    all_values = [iou for ious in iou_by_sl.values() for iou in ious]
    print("full mean", np.mean(all_values))
    all_means = [np.mean(ious) for ious in iou_by_sl.values()]
    print("sl mean", np.sum(all_means)/58)
