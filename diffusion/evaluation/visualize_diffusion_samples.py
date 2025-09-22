import os
from pathlib import Path
import contrastive.contrastive_evaluation_util as mesh_evaluation_util
import torch
import paths
import utils.util as util
from geometry.geometry_data import GeometryLoader
from ..generate_diffusion_embeddings import DiffusionEncodingConfig
from .diffusion_evaluation_util import DiffusionSampler, DiffusionSamplingConfig
from utils.visualizer_util import DaVinciVisualizer
from OCC.Extend.DataExchange import write_stl_file
import trimesh
import h5py

from cadlib.visualize import vec2CADsolid
from cadlib.util import visualize_program
from custom_visualization_sets import cad_visualization_sets

def get_image_of_cad(out_shape, save_path, visualizer, text=None, pl=None, save_path_3d=None):
    if save_path is None:
        temp_save_name = "_temp.stl"
    else:
        temp_save_name = save_path[:-4] + "_temp.stl"
    write_stl_file(out_shape, temp_save_name,
                               mode="binary",
                               linear_deflection=0.001,
                               angular_deflection=0.1)

    # Load the stl and take a picture
    mesh = trimesh.load(temp_save_name)
    verts = mesh.vertices
    faces = mesh.faces

    if len(verts) == 0:
        return

    visualizer.plot_picture(verts=verts, faces=faces, save_path=save_path, style="cad", text=text, pl=pl, save_path_3d=save_path_3d)

    os.remove(temp_save_name)




def generate_and_plot(encoder_type, contrastive_model_name, contrastive_checkpoint, diffusion_checkpoint, cad_ids, dataset="GenCAD3D", save_subfolder="visualization"):
    save_3d = True

    num_retries = 32
    diffusion_prior_suffix = "batch2048/"

    embedding_config = DiffusionEncodingConfig(encoder_type, contrastive_model_name, contrastive_checkpoint, dataset="GenCAD3D") # This is the training dataset, not the inference dataset

    diffusion_training_config = DiffusionSamplingConfig(diffusion_prior_suffix, embedding_config,
                                                        diffusion_checkpoint=diffusion_checkpoint)

    diffusion_sampler = DiffusionSampler(embedding_config=embedding_config,
                                         diffusion_sampling_config=diffusion_training_config)

    with_normals = embedding_config.use_normals

    check_intersection = True

    # Setup visualization
    output_folder = paths.HOME_PATH + diffusion_training_config.training_save_dir + "visualize_diffusion_out/"
    Path(output_folder).mkdir(parents=True, exist_ok=True)


    geometry_loader = GeometryLoader(data_root=paths.DATA_PATH + dataset + "/",
                                     phase="test", geometry_subdir="meshes/",
                                     with_normals=with_normals, stl_directory="stls/")
    # if use_set is not None:
    #     cad_ids = cad_visualization_sets[use_set]
    # else:
    #     min_length = 15
    #     max_length = None
    #     cad_ids = geometry_loader.get_cad_ids_of_sequence_length(min_length, max_length)[:20]
    num_samples = 6

    ## This create the cloud conversion / cad embedding
    cloud_to_cad = mesh_evaluation_util.GeometryToCAD(contrastive_model_name=contrastive_model_name,
                                                      encoder_ckpt_num=contrastive_checkpoint,
                                                      encoder_type=encoder_type)

    visualizer = DaVinciVisualizer(geometry_loader)
    torch.seed()

    # Actually do the conversion
    for cad_id in cad_ids:
        output_subfolder = save_subfolder + "/"
        Path(output_folder + output_subfolder).mkdir(parents=True, exist_ok=True)

        save_name = cad_id[cad_id.rfind("/") + 1:]

        mesh_pic_name = output_folder + output_subfolder + save_name + "_mesh.png"
        expected_pic_name = output_folder + output_subfolder + save_name + "_expected.png"
        save_path_3d_folder = output_folder + output_subfolder + save_name + "/"
        print(mesh_pic_name)
        print("saving to", save_path_3d_folder)

        if "mesh" in contrastive_model_name:
            style = "mesh"
        elif "pcn" in contrastive_model_name:
            style = "pcn"
        else:
            style = "pc"

        # pl = visualizer.make_new_plot(headless=True, r=1, c=num_samples + 2)
        # pl.subplot(0, 1)
        visualizer.plot_picture(data_id=cad_id, style=style, text="Input", normals=style == "pcn", save_path=mesh_pic_name, save_path_3d=save_path_3d_folder + "input.gltf")

        # pl.subplot(0, 0)
        visualizer.plot_picture(data_id=cad_id, style="cad", text="Expected Output", use_case="default", save_path=expected_pic_name, save_path_3d=save_path_3d_folder + "expected.gltf")

        if save_3d:
            cad_vec_expected = geometry_loader.load_cad(data_id=cad_id, tensor=False, as_single_vec=True)
            visualize_program(cad_vec_expected, save_path_3d_folder + "expected.png", legend=True)
        # Load geometry
        geometry = mesh_evaluation_util.load_geometry_input(encoder_type=encoder_type, data_ids=[cad_id],
                                                            geometry_loader=geometry_loader)

        cloud_embedding = cloud_to_cad.encode_geometry(*geometry, generation=True, normalize=False)

        # Diffusion prior: Create multiple
        pic_names = [mesh_pic_name, expected_pic_name]
        cad_pic_name = lambda x: output_folder + output_subfolder + save_name + f"_cad{x}.png"
        samples = diffusion_sampler.sample_N_force_valid(cloud_embedding, num_samples=num_samples, self_intersection_check=check_intersection, num_retries=num_retries)
        for j in range(len(samples)):
            # pl.subplot(0, j + 2)
            cad_embedding = samples[j]
            try:
                cad_encoding = cloud_to_cad.cad_latent_to_encoding(cad_embedding.unsqueeze(0))[0]
                out_shape = vec2CADsolid(cad_encoding)
            except Exception as e:
                print(e)
                print("bad decoding")
                continue

            if out_shape is None:
                print("bad decoding")
                continue

            try:
                # print(cad_pic_name(j))
                get_image_of_cad(out_shape, text="", visualizer=visualizer, save_path=cad_pic_name(j), save_path_3d=save_path_3d_folder + str(j) + ".gltf")
            except Exception as e:
                print("image error", e)
                continue

            if save_3d:
                save_path_program = save_path_3d_folder + "/" + str(j) + ".png"
                visualize_program(cad_encoding, save_path_program, legend=True)

            pic_names.append(cad_pic_name(j))
            # save the vec file
            name = cad_id[5:] + "_" + str(j)
            cad_save_name = output_folder + output_subfolder + save_name + "/cad_" + str(j) + ".h5"
            paths.mkdir(cad_save_name)
            print("cad saved to:", cad_save_name)
            with h5py.File(cad_save_name, "w") as f:
                f.create_dataset("vec", data=cad_encoding)

        # pl.screenshot(output_folder + output_subfolder + save_name + ".png")
        # pl.close()
        # combine images and delete originals
        util.combine_images(image_paths=pic_names,
                            save_name=output_folder + output_subfolder + save_name + "_comb.png")

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Eval Pipeline")
    parser.add_argument("-encoder_type", "--encoder_type", type=str, required=False, default=None,
                        help="pc, pcn, mesh_feast")
    parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=False, default=None,
                        help="name of contrastive model")
    parser.add_argument("-contrastive_checkpoint", "--contrastive_checkpoint", type=str, required=False, default=None,
                        help="checkpoint of retrieval model")
    parser.add_argument("-diffusion_checkpoint", "--diffusion_checkpoint", type=str, required=False, default=None,
                        help="checkpoint of diffusion model")

    args = parser.parse_args()

    # if args.dataset == "GenCAD3D":
    for complexity in ["simple", "medium", "complex", "very complex"]:
        generate_and_plot(args.encoder_type, args.contrastive_model_name, args.contrastive_checkpoint,
                          args.diffusion_checkpoint, cad_ids=cad_visualization_sets[complexity], save_subfolder=complexity)
