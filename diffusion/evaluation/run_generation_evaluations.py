import subprocess
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
parser.add_argument("-start_from_step", "--start_from_step", type=int, required=False, default=0,
                    help="command to start from")
parser.add_argument("-end_at_step", "--end_at_step", type=int, required=False, default=2,
                    help="command to start from")
args = parser.parse_args()


encoder_type = args.encoder_type
contrastive_model_name = args.contrastive_model_name
checkpoint = args.contrastive_checkpoint
dataset = args.dataset
# save_suffix = args.save_suffix
diffusion_checkpoint = args.diffusion_checkpoint

save_suffix = "batch2048/"

device = "cuda"

encoder_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, checkpoint, dataset=dataset)
print("==" * 50)
print("==" * 50)

diffusion_config = DiffusionSamplingConfig(save_suffix, encoder_config, diffusion_checkpoint=diffusion_checkpoint)
src = diffusion_config.eval_samples_output_dir[:-1]



if args.start_from_step < 1:
    run_args = ["-encoder_type", encoder_type,
                "-contrastive_model_name", contrastive_model_name,
                "-checkpoint", str(checkpoint),
                "-diffusion_checkpoint", str(diffusion_checkpoint)]
    if dataset is not None:
        run_args += ["-dataset", dataset]
    subprocess.run(["python", "-m", "diffusion.evaluation.generate_eval_encoding_samples"] + run_args, check=True)

if args.start_from_step < 2 and args.end_at_step > 0:
    subprocess.run(["python", "-m", "diffusion.evaluation.collect_gen_pc"] + ["--src", src], check=True)

if args.start_from_step < 3 and args.end_at_step > 1:
    subprocess.run(["python", "-m", "diffusion.evaluation.evaluate_gen_torch"] + ["--src", src + "_pc"], check=True)
