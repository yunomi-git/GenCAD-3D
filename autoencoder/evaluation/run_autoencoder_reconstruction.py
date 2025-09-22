
# # Reconstruction
# generate_genreation_reconstruction
# evaluate_ae_acc
# evaluate_cd_acc
# show_ae_by_sl

import subprocess
import argparse
import paths
from autoencoder.config_base import AutoencoderPathConfig

parser = argparse.ArgumentParser(description="Full Eval Pipeline")
parser.add_argument("-name", "--exp_name", type=str, required=False, default=None, help="path starting from content root")
parser.add_argument("-subfolder", "--subfolder", type=str, required=False, default=None, help="path starting from content root")
parser.add_argument("-ckpt", "--checkpoint", type=str, required=False, default="latest", help="path starting from content root")
parser.add_argument("-start_from_step", "--start_from_step", type=int, required=False, default=0,
                    help="command to start from")
parser.add_argument("-end_at_step", "--end_at_step", type=int, required=False, default=1,
                    help="command to start from")
parser.add_argument("-cd", "--cd", action='store_true')
parser.add_argument("-iou", "--iou", action='store_true')
parser.add_argument("-gen_step", "--iou_generate_step", action='store_true')

args = parser.parse_args()

exp_name = args.exp_name
save_dir = args.subfolder
checkpoint = args.checkpoint

config = AutoencoderPathConfig(exp_name)
encodings_source = config.exp_dir + "/eval/"+ save_dir + "/reconstructions"

print("==" * 50)
print("==" * 50)


if args.start_from_step <= 0:
    run_args = ["-name", exp_name,  # autoencoder_balanced_NA_FT_lr6_eval300
                "-ckpt", str(checkpoint),
                "--subfolder", save_dir,
                "--eval"
                ]
    subprocess.run(["python", "-m", "autoencoder.gencad.test_gencad"] + run_args, check=True)

subprocess.run(["python", "-m", "autoencoder.evaluation.evaluate_ae_acc"] + ["--src", paths.HOME_PATH + encodings_source], check=True)

if args.cd:
    subprocess.run(["python", "-m", "autoencoder.evaluation.evaluate_ae_cd"] + ["--src", paths.HOME_PATH + encodings_source], check=True)

if args.iou:
    run_args = ["--reconstruction_source", encodings_source + "/"]
    if args.iou_generate_step:
        run_args += ["--gen_step"]
    subprocess.run(["python", "-m", "autoencoder.evaluation.evaluate_IOU"] + run_args, check=True)
#
