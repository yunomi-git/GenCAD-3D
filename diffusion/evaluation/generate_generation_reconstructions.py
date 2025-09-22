import numpy as np
from ..generate_diffusion_embeddings import DiffusionEncodingConfig
from .diffusion_evaluation_util import DiffusionSampler, DiffusionSamplingConfig
from contrastive.contrastive_evaluation_util import GeometryToCAD
import h5py
from tqdm import tqdm
from cadlib.macro import (
    EOS_IDX
)
import os
from pathlib import Path
import paths
from geometry.geometry_data import GeometryLoader
import contrastive.contrastive_evaluation_util as mesh_evaluation_util
import argparse


# Afterwards use evaluate_ae_acc to assess performance.

parser = argparse.ArgumentParser(description="Full Eval Pipeline")
parser.add_argument("-encoder_type", "--encoder_type", type=str, required=False, default=None, help="pc, pcn, mesh_feast")
parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=False, default=None, help="path starting from content root")
parser.add_argument("-checkpoint", "--checkpoint", type=str, required=False, default=None, help="checkpoint of retrieval model")
parser.add_argument("-dataset", "--dataset", type=str, required=False, default=None,
                    help="name of the dataset to encode")
parser.add_argument("-diffusion_checkpoint", "--diffusion_checkpoint", type=str, required=False, default=None,
                    help="checkpoint of diffusion model")
args = parser.parse_args()

if __name__=="__main__":
    # Diffusion Model Hyperparameters
    device = "cuda"


    encoder_type = args.encoder_type
    contrastive_model_name = args.contrastive_model_name
    ckpt = args.checkpoint
    dataset = args.dataset
    # save_suffix = args.save_suffix
    diffusion_checkpoint = args.diffusion_checkpoint


    encoder_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, ckpt, dataset=dataset)

    save_suffix = "batch2048/"
    batch_size = 128

    # Grab Dataloader
    data_root = paths.DATA_PATH + "GenCAD3D/"

    encoder_args = mesh_evaluation_util.get_geometry_encoder_args(contrastive_model_name)
    geometry_loader = GeometryLoader(data_root=data_root, with_normals=encoder_args["use_normals"],
                                                                 geometry_subdir="meshes/", phase="test")

    # Embeddings
    cache_folder = paths.HOME_PATH + "visualization/embedding_cache/"

    # The geometry embeddings do not change when generation is set to true/false
    contrastive_model_name_name = contrastive_model_name[contrastive_model_name.find("/")+1:-1]
    embedding_loader = mesh_evaluation_util.GeometryEmbeddingSpace(encoder_type=encoder_type, num_geometries=None,
                                              cache_parent_dir=cache_folder, geometry_loader=geometry_loader,
                                                                   generation=False,
                                              ckpt_name=contrastive_model_name_name + str(ckpt))

    # geometry encoder and cad decoder
    geometry_to_cad = GeometryToCAD(encoder_type=encoder_type, contrastive_model_name=contrastive_model_name, encoder_ckpt_num=ckpt)

    geometry_embeddings, space_cad_ids = embedding_loader.load_geometry_space_embeddings(geometry_to_cad=geometry_to_cad, normalize=False)

    # Diffusion Model
    diffusion_config = DiffusionSamplingConfig(save_suffix, encoder_config, diffusion_checkpoint=diffusion_checkpoint)

    diffusion_inference = DiffusionSampler(embedding_config=encoder_config,
                                           diffusion_sampling_config=diffusion_config)



    # We want to open up the embeddings of the geometries, grab the test set, and inference
    Path(paths.HOME_PATH + diffusion_config.eval_reconstruction_output_dir).mkdir(parents=True, exist_ok=True)

    from utils.util import DictList
    sl_holder = DictList()
    for i in tqdm(range(int(np.ceil(len(geometry_embeddings) / batch_size)))):
        geometry_embedding_batch = geometry_embeddings[batch_size * i : batch_size * (i+1)]
        cad_ids_batch = space_cad_ids[batch_size * i : batch_size * (i+1)]
        batch_cad_embedding = diffusion_inference.sample(geometry_embedding_batch)
        batch_cad_encoding = geometry_to_cad.cad_latent_to_encoding(batch_cad_embedding)

        batch_gt_commands, batch_gt_args = geometry_loader.load_cad_batch(cad_ids_batch)

        # Validate each embedding individually
        # Save each encoding in a separate file
        for k in range(len(cad_ids_batch)):
            cad_embedding = batch_cad_embedding[k]
            out_vec = batch_cad_encoding[k]

            gt_command = batch_gt_commands[k].detach().cpu().numpy()
            gt_args = batch_gt_args[k].detach().cpu().numpy()
            gt_vec = np.hstack([gt_command[:, np.newaxis], gt_args])
            cad_id = cad_ids_batch[k]

            # Grab sequence length
            # out_command = out_vec[:, 0]
            # seq_len = out_command.tolist().index(EOS_IDX)
            # gt_command = gt_vec[:, 0]
            seq_len = gt_command.tolist().index(EOS_IDX)
            sl_holder.add_to_key(seq_len, 1)

            # cad_id_name = cad_id[cad_id.find("/")]
            save_path = paths.HOME_PATH + os.path.join(diffusion_config.eval_reconstruction_output_dir, f'{cad_id[cad_id.find("/")+1:]}.h5')
            # Path(save_path[:save_path.rfind("/")]).mkdir(parents=True, exist_ok=True)

            with h5py.File(save_path, 'w') as fp:
                phase = "test"
                fp.create_dataset('{}_zs'.format(phase), data=cad_embedding.detach().cpu().numpy())
                fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=int)
                fp.create_dataset('gt_vec', data=gt_vec[:seq_len], dtype=int)

                # fp.create_dataset('cad_id', data=, dtype=int)

            # sample_num += 1

    print(str(sl_holder))
    print(f'--------- data saved at: {diffusion_config.eval_samples_output_dir}')
    print('# ' * 20)

