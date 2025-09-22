
# # Reconstruction
# generate_genreation_reconstruction
# evaluate_ae_acc
# evaluate_cd_acc
# show_ae_by_sl

import subprocess

import paths
from .diffusion_evaluation_util import DiffusionSamplingConfig
from ..generate_diffusion_embeddings import DiffusionEncodingConfig
import argparse




parser = argparse.ArgumentParser(description="Full Eval Pipeline")
parser.add_argument("-encoder_type", "--encoder_type", type=str, required=False, default=None, help="pc, pcn, mesh_feast")
parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=False, default=None, help="name of contrastive model")
parser.add_argument("-contrastive_checkpoint", "--contrastive_checkpoint", type=str, required=False, default=None, help="checkpoint of retrieval model")
parser.add_argument("-dataset", "--dataset", type=str, required=False, default="GenCAD3D",
                    help="name of the dataset to encode")
parser.add_argument("-diffusion_checkpoint", "--diffusion_checkpoint", type=str, required=False, default=None,
                    help="checkpoint of diffusion model")
parser.add_argument("-skip_sample", "--skip_sample_generation", action="store_true")
parser.add_argument("-cd", "--cd", action="store_true")
parser.add_argument("-iou", "--iou", action="store_true")
parser.add_argument("-gen_step", "--iou_generate_step", action="store_true")


args = parser.parse_args()


encoder_type = args.encoder_type
contrastive_model_name = args.contrastive_model_name
contrastive_checkpoint = args.contrastive_checkpoint
dataset = args.dataset
# save_suffix = args.save_suffix
diffusion_checkpoint = args.diffusion_checkpoint

save_suffix = "batch2048/"

device = "cuda"

encoder_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, contrastive_checkpoint, dataset=dataset)
print("==" * 50)
print("==" * 50)

diffusion_config = DiffusionSamplingConfig(save_suffix, encoder_config, diffusion_checkpoint=diffusion_checkpoint)
src = diffusion_config.eval_reconstruction_output_dir[:-1]


if not args.skip_sample_generation:
    run_args = ["-encoder_type", encoder_type,
                "-contrastive_model_name", contrastive_model_name,
                "-checkpoint", str(contrastive_checkpoint),
                "-diffusion_checkpoint", str(diffusion_checkpoint)]
    if dataset is not None:
        run_args += ["-dataset", dataset]
    subprocess.run(["python", "-m", "diffusion.evaluation.generate_generation_reconstructions"] + run_args, check=True)

subprocess.run(["python", "-m", "autoencoder.evaluation.evaluate_ae_acc"] + ["--src", paths.HOME_PATH + src], check=True)

if args.cd:
    subprocess.run(["python", "-m", "autoencoder.evaluation.evaluate_ae_cd"] + ["--src", paths.HOME_PATH + src], check=True)


if args.iou:
    run_args = ["--reconstruction_source", src + "/"]
    if args.iou_generate_step:
        run_args += ["--gen_step"]
    subprocess.run(["python", "-m", "autoencoder.evaluation.evaluate_IOU"] + run_args, check=True)
#