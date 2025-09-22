import numpy as np
from ..generate_diffusion_embeddings import DiffusionEncodingConfig
from .diffusion_evaluation_util import DiffusionSampler, DiffusionSamplingConfig
from contrastive.contrastive_evaluation_util import GeometryToCAD
import h5py
from tqdm import tqdm
from cadlib.macro import EOS_IDX
import os
from pathlib import Path
from cadlib.visualize import vec2CADsolid
import paths
import argparse

parser = argparse.ArgumentParser(description="Full Eval Pipeline")
parser.add_argument("-encoder_type", "--encoder_type", type=str, required=False, default=None, help="pc, pcn, mesh_feast")
parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=False, default=None, help="path starting from content root")
parser.add_argument("-checkpoint", "--checkpoint", type=str, required=False, default=None, help="checkpoint of retrieval model")
parser.add_argument("-dataset", "--dataset", type=str, required=False, default=None,
                    help="name of the dataset to encode")
parser.add_argument("-diffusion_checkpoint", "--diffusion_checkpoint", type=str, required=False, default=40,
                    help="checkpoint of diffusion model")
args = parser.parse_args()

if __name__=="__main__":
    device = "cuda"

    encoder_type = args.encoder_type
    contrastive_model_name = args.contrastive_model_name
    checkpoint = args.checkpoint
    dataset = args.dataset
    # save_suffix = args.save_suffix
    diffusion_checkpoint = args.diffusion_checkpoint

    save_suffix = "batch2048/"

    encoder_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, checkpoint, dataset=dataset)

    attempt_force_valid = False


    # CAD decoder
    geometry_to_cad = GeometryToCAD(encoder_type=encoder_type, contrastive_model_name=contrastive_model_name, encoder_ckpt_num=checkpoint)

    # Diffusion model
    diffusion_config = DiffusionSamplingConfig(save_suffix, encoder_config, diffusion_checkpoint=diffusion_checkpoint)

    diffusion_inference = DiffusionSampler(embedding_config=encoder_config,
                                           diffusion_sampling_config=diffusion_config)


    print("generating from ", diffusion_config.eval_samples_output_dir)

    # We want to open up the embeddings of the geometries, grab the test set, and inference
    Path(paths.HOME_PATH + diffusion_config.eval_samples_output_dir).mkdir(parents=True, exist_ok=True)

    batch_size = 128
    geometry_latents = diffusion_inference.geometry_latents
    num_initial_invalid = 0
    sample_num = 0
    num_bad = 0
    total_generated = 0
    for i in tqdm(range(int(np.ceil(len(geometry_latents) / batch_size)))):
        geometry_embedding_batch = geometry_latents[batch_size * i : batch_size * (i+1)]
        batch_cad_embedding = diffusion_inference.sample(geometry_embedding_batch)
        batch_cad_encoding = geometry_to_cad.cad_latent_to_encoding(batch_cad_embedding)

        # Validate each embedding individually
        # Save each encoding in a separate file
        for k in range(len(batch_cad_encoding)):
            cad_embedding = batch_cad_embedding[k]
            out_vec = batch_cad_encoding[k]
            try:
                shape = vec2CADsolid(out_vec)
            except:
                print("Error: Regenerating")
                num_initial_invalid += 1
                if attempt_force_valid:
                    try:
                        geometry_embedding = geometry_embedding_batch[k].unsqueeze(0)
                        cad_embedding = diffusion_inference.sample_single_force_valid(geometry_embedding)
                        out_vec = geometry_to_cad.cad_latent_to_encoding(cad_embedding.unsqueeze(0))[0]
                        shape = vec2CADsolid(out_vec)
                    except:
                        num_bad += 1
                        print(num_bad)
                        continue

            # Grab sequence length
            out_command = out_vec[:, 0]
            try:
                seq_len = out_command.tolist().index(EOS_IDX)
            except:
                print("Error: Regenerating")
                num_initial_invalid += 1

            save_path = paths.HOME_PATH + os.path.join(diffusion_config.eval_samples_output_dir, f'sample_{sample_num}.h5')
            with h5py.File(save_path, 'w') as fp:
                phase = "test"
                fp.create_dataset('{}_zs'.format(phase), data=cad_embedding.detach().cpu().numpy())
                fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=int)
                fp.create_dataset('seq_len', data=seq_len, dtype=int)

                # fp.create_dataset('cad_id', data=, dtype=int)

            sample_num += 1

    print("num_skipped", num_bad, "total generated", sample_num + 1, "expected total", len(geometry_latents), "initial invalid ratio", num_initial_invalid / (sample_num + 1))
    print(f'--------- data saved at: {diffusion_config.eval_samples_output_dir}')
    print('# ' * 20)

