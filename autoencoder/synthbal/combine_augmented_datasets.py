import paths
from cadlib.macro import AVAILABLE_SEQUENCE_LENGTHS
from paths import DirectoryPathManager
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

if __name__=="__main__":
    # Given datapath/save_name_TEMP/ as the output of generate_augmented_dataset,
    # combines them into datapath/save_name/ as the final dataset.
    # You can delete datapath/save_name_TEMP/
    # Follow with generate_splits
    parser = argparse.ArgumentParser(description="Combine separate synthetic data.")
    parser.add_argument("-name", "--save_name", type=str, default="GenCAD3D_SynthBal", required=False)
    args = parser.parse_args()

    new_folder_name = paths.DATA_PATH + args.save_name + "/"
    Path(new_folder_name).mkdir(parents=True, exist_ok=True)

    orig_name_base = paths.DATA_PATH + args.save_name + "_TEMP/"

    for sequence_length in tqdm(AVAILABLE_SEQUENCE_LENGTHS):
        folder_base = orig_name_base + str(sequence_length) + "/"
        path_manager = DirectoryPathManager(base_path=folder_base + "/", base_unit_is_file=True)

        all_files_relative = path_manager.get_files_relative(extension=True)
        # if len(all_files_relative) < 2000:
        #     print(sequence_length)

        for file in all_files_relative:
            new_file_name = new_folder_name + file
            new_folder = new_folder_name + file[:file.rfind("/")]
            Path(new_folder).mkdir(parents=True, exist_ok=True)

            shutil.copyfile(src=folder_base + file, dst=new_folder_name + file)

