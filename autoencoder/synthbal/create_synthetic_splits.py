import os

import paths
import sklearn.model_selection
import json
import glob
import random
from cadlib.macro import AVAILABLE_SEQUENCE_LENGTHS
from geometry.geometry_data import GeometryLoader
from utils.util import DictList
from tqdm import tqdm
import argparse

def sequence_length_filter_from_basis():
    # Looks at a reference dataset split.
    # For each phase, grab N items from each sequence length and get an equal number of each

    data_root = paths.DATA_PATH + "GenCAD3D/"
    reference_splits_name = data_root + "filtered_data.json"

    with open(reference_splits_name, 'r') as f:
        reference_splits = json.load(f)

    new_splits = {}

    for phase in ["train", "test", "validation"]:
        geometry_loader = GeometryLoader(data_root, phase=phase)
        sl_dict = {}
        min_length = -1
        max_length = 0
        for sequence_length in AVAILABLE_SEQUENCE_LENGTHS:
            sl_dict[sequence_length] = geometry_loader.get_cad_ids_of_sequence_length(min_length=sequence_length, max_length=sequence_length, cad=True)
            if min_length < 0 or len(sl_dict[sequence_length]) < min_length:
                min_length = len(sl_dict[sequence_length])
            if len(sl_dict[sequence_length]) > max_length:
                max_length = len(sl_dict[sequence_length])

        print(phase, "min/max", min_length, max_length)
        new_data = []

        for sequence_length in AVAILABLE_SEQUENCE_LENGTHS:
            new_data += random.sample(sl_dict[sequence_length], min_length)
        new_splits[phase] = new_data

    with open(data_root + "filtered_data_balanced.json", 'w') as f:
        json.dump(new_splits, f)
    print("")

    num_items_in_splits = 0
    num_items_in_orig = 0
    for phase in ["train", "test", "validation"]:
        num_items_in_splits += len(new_splits[phase])
        num_items_in_orig += len(reference_splits[phase])
        print(phase, ":", "new", len(new_splits[phase]), "| old", len(reference_splits[phase]))
    print("length of splits", num_items_in_splits)
    print("num items in original", num_items_in_orig)

def from_basis(basis_dataset, new_dataset):
    # Looks at a reference dataset split.
    # For each phase, for each item in reference dataset, finds similar items in new dataset and moves it to the corresponding phase
    # This will prevent data from leaking between phases between the new and old dataset
    data_root = paths.DATA_PATH + new_dataset + "/"
    reference_splits_name = paths.DATA_PATH + basis_dataset + "/filtered_data.json"

    with open(reference_splits_name, 'r') as f:
        reference_splits = json.load(f)

    director_manager = paths.DirectoryPathManager(base_path=data_root + "cad_vec/", base_unit_is_file=True)
    all_files = director_manager.get_files_relative(extension=False)

    # Creates an association dictionary
    print("Creating lookup table")
    file_lookup = DictList()
    for file in tqdm(all_files, smoothing=0.1):
        # key = file[:file.rfind("/")]
        if "_" in file:
            key = file[:file.find("_")]
        else:
            key = file
        file_lookup.add_to_key(key, file)


    new_splits = {}

    for phase in ["train", "test", "validation"]:
        new_data = []
        original_data = reference_splits[phase]

        for data_id in original_data:
            if file_lookup.key_exists(data_id):
                new_data += file_lookup.get_key(data_id)

        new_splits[phase] = new_data

    with open(data_root + "filtered_data.json", 'w') as f:
        json.dump(new_splits, f)
    print("")

    num_items_in_splits = 0
    num_items_in_orig = 0
    for phase in ["train", "test", "validation"]:
        num_items_in_splits += len(new_splits[phase])
        num_items_in_orig += len(reference_splits[phase])
        print(phase, ":", "new", len(new_splits[phase]), "| old", len(reference_splits[phase]))
    print("all data found", len(all_files), "length of splits", num_items_in_splits)
    print("num items in original", num_items_in_orig)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create splits for the new data based on old dataset.")
    # This is the reference dataset
    parser.add_argument("-input_dataset", "--input_dataset", type=str, default="GenCAD3D", required=False)
    # This is the new dataset. The splits should match the reference dataset so that data doesnt leak from original_test to synthetic_train
    parser.add_argument("-name", "--save_name", type=str, default="GenCAD3D_SynthBal", required=True)
    args = parser.parse_args()

    from_basis(args.input_dataset, args.save_name)
    # sequence_length_filter_from_basis()
