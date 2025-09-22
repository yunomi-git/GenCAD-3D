import numpy as np

import paths
from geometry.geometry_data import GeometryLoader
from pathlib import Path
import random
import h5py
from autoencoder.synthbal.augmentations import Augmentations
from cadlib.visualize import vec2CADsolid_valid_check
from tqdm import tqdm
from cadlib.macro import (AVAILABLE_SEQUENCE_LENGTHS)
import faulthandler
import json

# Follow with combine_augmented_datasets and create_filtered_data

faulthandler.enable()

def save(dataset_dir, data_id, data):
    save_path = dataset_dir + data_id + ".h5"
    sub_dir = data_id[:data_id.find("/")]
    Path(dataset_dir + sub_dir).mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset('vec', data=data)

class GenerationTemplate:
    def __init__(self, augmentations, add_small_noise=True, error_check=True):
        self.augmentations = augmentations
        self.add_small_noise = add_small_noise
        self.error_check = error_check

settings = {
    "LN_snRS": GenerationTemplate({
        'replace_sketch': 0.6,
        'noise': 0.4,
        }, True),
    "SN_snRS": GenerationTemplate({
        'replace_sketch': 0.6,
        'small_noise': 0.4,
        }, True),
    "snRS": GenerationTemplate({
        'replace_sketch': 1.0,
        }, True),
    "RS": GenerationTemplate({
        'replace_sketch': 1.0,
        }, False),
    "SN": GenerationTemplate({
        'small_noise': 1.0,
        }, False),
    "SN_RS": GenerationTemplate({
        'replace_sketch': 0.6,
        'small_noise': 0.4,
        }, False),

    "LN_snRS_ne": GenerationTemplate({
        'replace_sketch': 0.6,
        'noise': 0.4,
        }, True, False),
    "RRE_ne": GenerationTemplate({
        'rre': 1.0,
    }, False, False),
    "LN_snRRE_ne": GenerationTemplate({
        'rre': 0.6,
        'noise': 0.4,
    }, True, False),

    "RRE": GenerationTemplate({
         'rre': 1.0,
     }, False),
    "LN_snRRE": GenerationTemplate({
        'rre': 0.6,
        'noise': 0.4,
        }, True),
    "LN_snReE": GenerationTemplate({
        're-extrude': 0.6,
        'noise': 0.4,
        }, True),
    "LN_snRE": GenerationTemplate({
        'replace_extrusion': 0.6,
        'noise': 0.4,
        }, True),
    "LN_snArc": GenerationTemplate({
            'arc2': 0.6,
            'noise': 0.4,
        }, True),
}

def augment(cad_vec, augmentation_manager, template=None):
    # augment on unpadded, concatenated cad_vec
    if template is None: # This is default setting
        augmentations = {
            'replace_sketch': 0.6,
            'noise': 0.4,
        }
        add_noise_to_aug = True
    else:
        augmentations = settings[template].augmentations
        add_noise_to_aug = settings[template].add_small_noise

    augment_type = random.choices(list(augmentations.keys()), k=1, cum_weights=list(augmentations.values()))[0]

    if add_noise_to_aug and 'noise' not in augment_type:
        cad_vec = augmentation_manager.dataset_augment(cad_vec, "small_noise")

    cad_vec = augmentation_manager.dataset_augment(cad_vec, augment_type)

    return cad_vec


import argparse


if __name__=="__main__":
    # Given input dataset datadir/input_dataset/,
    # Creates a temporary dataset datadir/save_name_TEMP/
    # This creates separate subfolders for each sequence length.
    # Postprocess with combine_augmented_datasets.
    parser = argparse.ArgumentParser(description="Generate Synthetic Data.")
    parser.add_argument("-min", "--min", type=int, default=3, required=False)
    parser.add_argument("-max", "--max", type=int, default=59, required=False)
    parser.add_argument("-input_dataset", "--input_dataset", type=str, default="GenCAD3D", required=False)
    parser.add_argument("-name", "--save_name", type=str, default="GenCAD3D_SynthBal", required=False)
    parser.add_argument("-template", "--template", type=str, default=None, required=False)
    parser.add_argument("-total_desired_datapoints", "--total_desired_datapoints", type=int, default=170000, required=False)

    args = parser.parse_args()
    template = args.template

    total_desired_datapoints = args.total_desired_datapoints
    populate_with_existing_percent = 0.2
    if template is None:
        new_dataset_name = args.save_name
        do_error_check = True
    else:
        new_dataset_name = "DaVinci_CAD_" + template
        do_error_check = settings[template].error_check

    data_root = paths.DATA_PATH + args.input_dataset + "/"


    orig_splits_name = data_root + "filtered_data.json"
    with open(orig_splits_name, 'r') as f:
        orig_splits = json.load(f)

    total_orig_datapoints = np.sum([len(orig_splits[phase]) for phase in orig_splits.keys()])
    desired_numbers = {phase: int(len(orig_splits[phase]) * total_desired_datapoints / total_orig_datapoints) for phase in orig_splits.keys()}

    min_val = int(args.min)
    max_val = int(args.max)
    sequence_lengths = list(range(min_val, max_val + 1))
    sequence_lengths.reverse()

    dataloaders = {}
    augmentation_managers = {}
    for phase in ["train", "validation", "test"]:
        dataloaders[phase] = GeometryLoader(data_root=data_root, phase=phase)
        dataloaders[phase].get_all_valid_data(cad=True, mesh=False, cloud=False)
        augmentation_managers[phase] = Augmentations(data_root, phase=phase)

    for seq_length in sequence_lengths:
        new_dataset_dir = paths.DATA_PATH + new_dataset_name + "_TEMP/" + str(seq_length) + "/cad_vec/"
        Path(new_dataset_dir).mkdir(parents=True, exist_ok=True)
        print("========== seq len", seq_length)

        for phase in ["train", "validation", "test"]:
            total_desired_cad_models = desired_numbers[phase]
            print("===== Generating for ", phase, "set: ", total_desired_cad_models)

            # augmentation_manager = Augmentations(data_root, phase=phase)
            geometry_loader = dataloaders[phase]
            augmentation_manager = augmentation_managers[phase]

            num_cad_per_seq_length = total_desired_cad_models // len(AVAILABLE_SEQUENCE_LENGTHS)



            original_data_ids = geometry_loader.get_cad_ids_of_sequence_length(min_length=seq_length, max_length=seq_length,
                                                                               cad=True, mesh=False, cloud=False)

            num_existing_desired = int(num_cad_per_seq_length * populate_with_existing_percent)
            num_existing_desired = min(num_existing_desired, len(original_data_ids))

            # save as many existing as possible
            original_ids_to_transfer = random.sample(original_data_ids, num_existing_desired)

            for data_id in original_ids_to_transfer:
                command, args = geometry_loader.load_cad(data_id, tensor=False, pad=False)
                cad_vec = np.hstack([command[:, np.newaxis], args])
                save(new_dataset_dir, data_id, cad_vec)

            print("num existing moved", len(original_ids_to_transfer))

            # then save the debt
            num_augmentations_needed = num_cad_per_seq_length - num_existing_desired
            index = 0
            pbar = tqdm(total=num_augmentations_needed, position=0, leave=True)
            errors = 0
            print("num synthbal needed", num_augmentations_needed)
            while num_augmentations_needed > 0:
                data_id = random.choice(original_data_ids)
                command, args = geometry_loader.load_cad(data_id, tensor=False, pad=False)
                cad_vec = np.hstack([command[:, np.newaxis], args])

                augmented_cad_vec = augment(cad_vec, augmentation_manager, template)

                pbar.set_description(f"ID {data_id} | Num Errors {errors}")
                if do_error_check:
                    shape = vec2CADsolid_valid_check(augmented_cad_vec)
                    if shape is None:
                        errors += 1
                        pbar.set_description(f"ID {data_id} | Num Errors {errors}")
                        continue

                # save the augmentation
                augmentation_data_id = data_id + "_" + str(index)
                save(new_dataset_dir, augmentation_data_id, augmented_cad_vec)

                num_augmentations_needed -= 1
                index += 1
                pbar.update(1)
            print("errors", errors)

