import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import paths
from contrastive.contrastive_evaluation_util import GeometryEmbeddingSpace
import torch
from pathlib import Path
import utils.util as util
import contrastive.contrastive_evaluation_util as mesh_evaluation_util
import random
from geometry.geometry_data import GeometryLoader
from utils.visualizer_util import DaVinciVisualizer
from .. import batch_creator
from cadlib.util import visualize_program
from custom_visualization_sets import cad_visualization_sets

device = "cuda:0"

def find_closest_embedding(search_space_embeddings, query_embedding):
    # given: cad search space embeddings, mesh embedding
    # find for mesh: closest n cads (as indices)
    loss = torch.nn.CosineSimilarity()
    probabilities = loss(query_embedding, search_space_embeddings)

    sorted_indices = torch.argsort(probabilities, descending=True).cpu()
    # num_best = 10
    # top_n_indices = sorted_indices[:num_best]
    return sorted_indices, probabilities

def do_randomized_experiment(num_values, query_index):
    probabilities = torch.zeros(num_values)
    sorted_indices = torch.randperm(num_values)

    ranking = np.where((sorted_indices == query_index).numpy())[0][0]
    confidence = probabilities[query_index].item()
    return ranking, confidence

def do_experiment(geometry_space_embeddings, space_cad_ids, test_name=None, query_index=None, cad_space_embeddings=None,
                  geometry_loader=None, output_folder=None, visualize=False, encoder_type="mesh", save_3d=False):
    retrieval_space = geometry_space_embeddings
    if cad_space_embeddings is not None:
        retrieval_space = cad_space_embeddings

    # if input_cloud exists, this is a pre_computed experiment
    # otherwise, choose inputs randomly
    if query_index is None:
        test_inds = random.choices(list(range(len(space_cad_ids))), k=1)
        query_index = test_inds[0]

    embedding_source1 = geometry_space_embeddings[query_index]

    sorted_indices, probabilities = find_closest_embedding(query_embedding=embedding_source1, search_space_embeddings=retrieval_space)

    ranking = np.where((sorted_indices == query_index).numpy())[0][0]
    confidence = probabilities[query_index].item()

    if visualize:
        visualizer = DaVinciVisualizer(geometry_loader=geometry_loader)
        save_path_3d_folder = None
        if save_3d:
            cad_id = space_cad_ids[query_index]
            save_folder = cad_id[cad_id.rfind("/") + 1:]
            save_path_3d_folder = output_folder + save_folder + "/"
            paths.mkdir(save_path_3d_folder)
            print("saving 3d", save_path_3d_folder)

        visualizer.plot_picture(data_id=space_cad_ids[query_index], style=encoder_type, text="Input",
                                save_path=output_folder + "source1" + "_mesh.png",
                                normals=encoder_type=="pcn", save_path_3d=save_path_3d_folder + "input.gltf")


        # Save other pics
        num_best = 6
        top_n_indices = sorted_indices[:num_best]
        top_n_indices = np.insert(top_n_indices, 0, query_index, axis=0)
        top_n_pics = []
        for i in range(num_best):
            name = "pred_output_" + str(i)
            probability = probabilities[top_n_indices[i]].item()
            probability = round(probability, 3)

            if i == 0: # First one is the expected
                text = "Expected"
            else:
                text = "p=" + str(probability)

            save_path_3d = None
            if save_3d:
                cad_id = space_cad_ids[top_n_indices[i]]
                save_name = str(i-1)
                if i == 0:
                    save_name = "expected"
                save_path_3d = save_path_3d_folder + save_name + ".gltf"
                cad_vec_expected = geometry_loader.load_cad(data_id=cad_id, tensor=False, as_single_vec=True)
                visualize_program(cad_vec_expected, save_path_3d_folder + save_name + ".png", legend=True)

            visualizer.plot_picture(data_id=space_cad_ids[top_n_indices[i]], style="cad", text=text,
                                    save_path=output_folder + name + "_mesh.png", use_case="default", save_path_3d=save_path_3d)
            top_n_pics.append(output_folder + name + "_mesh.png")


        image_paths = [output_folder + "source1_mesh.png"]
        image_paths.extend(top_n_pics)
        util.combine_images(image_paths=image_paths,
                                        save_name=output_folder + test_name + "_outputs.png")


    return ranking, confidence

class RankHolder:
    def __init__(self, rank_categories, categories_are_percents, num_in_space):
        self.rank_holder = {}
        for rank in rank_categories:
            self.rank_holder[rank] = 0

        self.use_percents = categories_are_percents
        self.rank_categories = rank_categories

        self.num_in_space = num_in_space

        if self.use_percents:
            self.ranking_function = lambda rank, category: rank * 1.0 / self.num_in_space < category / 100.0
        else:
            self.ranking_function = lambda rank, category: rank < category

    def add_rank(self, rank):
        for rank_cat in self.rank_categories:
            if self.ranking_function(rank, rank_cat):
                self.rank_holder[rank_cat] += 1

    def print_scores(self, num_experiments):
        print("\t\t\t", "num", "\t", "% of " + str(num_experiments) + " retrievals")
        for rank_cat in self.rank_categories:
            score = self.rank_holder[rank_cat]
            top_suffix = ""
            if self.use_percents:
                top_suffix = "%"
            print("top-" + str(rank_cat) + top_suffix + "   ", "\t", score, "\t", score * 100.0 / num_experiments)


def evaluate_random_retrieval(ranking_categories, num_geometries=4096, print_stats=True):
    num_exp = num_geometries
    # Choose combinations within these embeddings
    total_ranking = 0
    total_confidence = 0

    rank_holder = RankHolder(ranking_categories, categories_are_percents=False, num_in_space=num_geometries)
    rank_holder_per = RankHolder(ranking_categories, categories_are_percents=True, num_in_space=num_geometries)

    for i in range(num_exp):
        ranking, confidence = do_randomized_experiment(num_values=num_geometries, query_index=i)

        total_ranking += ranking
        total_confidence += confidence
        rank_holder.add_rank(ranking)
        rank_holder_per.add_rank(ranking)


    if print_stats:
        print("====")
        print("Encoder: Random")
        print("Retrieval from", num_geometries, "geometries/cad")
        print("-")
        rank_holder_per.print_scores(num_exp)
        print("-")
        rank_holder.print_scores(num_exp)

        print("avg_ranking", total_ranking / num_exp, "avg_confidence", total_confidence / num_exp)
    return rank_holder, rank_holder_per

def create_classes(ckpt, encoder_type, contrastive_model_name, num_geometries, use_augmented=False, min_seq_len=None, max_seq_len=None, normalize=False, geom_subdir=None):
    geometry_to_cad = mesh_evaluation_util.GeometryToCAD(encoder_type=encoder_type,
                                                         contrastive_model_name=contrastive_model_name,
                                                         encoder_ckpt_num=ckpt)
    data_root = paths.DATA_PATH + "GenCAD3D/"

    if geom_subdir is None:
        geom_subdir = "meshes/"

    geometry_loader = GeometryLoader(data_root=data_root, phase="test",
                                     with_normals=geometry_to_cad.use_normals,
                                     geometry_subdir=geom_subdir,
                                     stl_directory="stls/")

    cache_folder = "visualization/embedding_cache/"
    dataset_name = None

    embedding_loader = GeometryEmbeddingSpace(encoder_type=encoder_type, num_geometries=num_geometries,
                                              cache_parent_dir=cache_folder, geometry_loader=geometry_loader,
                                              ckpt_name=contrastive_model_name + str(ckpt), dataset_name=dataset_name,
                                              min_seq_len=min_seq_len, max_seq_len=max_seq_len, normalize=normalize)

    return geometry_to_cad, geometry_loader, embedding_loader

def evaluate_retrieval(ckpt, encoder_type, contrastive_model_name, ranking_categories, num_geometries=4096, requested_cad_ids=None,
                       randomize_subbatch=False, subfolder_name=None, print_stats=True, plot_first_n_retrievals=0,
                       geometry_to_cad=None, geometry_loader=None, embedding_loader=None, evaluate_on_augmented=False,
                       batch_loader:batch_creator.BatchCreator=None, vis_3d=False):
    ### Arguments:
    # if requested_cad_ids is set, then perform retrieval on those cad ids
    # If it is not set, then perform retrieval for each geometry in the entire space
    ###

    if geometry_to_cad is None or geometry_loader is None:
        geometry_to_cad, geometry_loader, embedding_loader = create_classes(ckpt, encoder_type, contrastive_model_name, num_geometries, use_augmented=evaluate_on_augmented)

    # pick n random files -> embedding space
    # given embedding v, find closest embedding in space

    output_folder = ("visualization/retrieval/" + encoder_type + "/"
                     + contrastive_model_name + "/"
                     + "ckpt-" + str(ckpt) + "/"
                     + str(num_geometries) + "/")
    if subfolder_name is not None:
        output_folder += subfolder_name + "/"

    if batch_loader is not None:
        space_cad_ids = batch_loader.get_batch(num_samples=num_geometries, randomize=randomize_subbatch)
        embedding_loader.set_specific_subbatch(space_cad_ids)
        geometry_embeddings, _ = embedding_loader.load_geometry_space_embeddings(
            geometry_to_cad=geometry_to_cad, normalize=False)
        cad_embeddings, _ = embedding_loader.load_cad_embeddings(geometry_to_cad=geometry_to_cad, normalize=False)

    else:
        if randomize_subbatch:
            embedding_loader.do_randomize_subbatch()

        geometry_embeddings, space_cad_ids = embedding_loader.load_geometry_space_embeddings(geometry_to_cad=geometry_to_cad, normalize=False)
        cad_embeddings, _ = embedding_loader.load_cad_embeddings(geometry_to_cad=geometry_to_cad, normalize=False)

    # cad_id_dict = embedding_loader.cad_id_to_idx
    space_cad_id_dict = {space_cad_ids[i]: i for i in range(len(space_cad_ids))}

    # Choose combinations within these embeddings
    total_ranking = 0
    total_confidence = 0

    rank_holder = RankHolder(ranking_categories, categories_are_percents=False, num_in_space=num_geometries)
    rank_holder_per = RankHolder(ranking_categories, categories_are_percents=True, num_in_space=num_geometries)

    # 2 cases: looking at specific cad, or looking at entire space
    # If requesting a specific id, add it into the space!
    if requested_cad_ids is not None:
        for i in range(len(requested_cad_ids)):
            if i < plot_first_n_retrievals:
                visualize = True
                Path(output_folder).mkdir(parents=True, exist_ok=True)
            else:
                visualize = False

            # For each cad id, add it in place 1 of the embedding space
            query_cad_id = requested_cad_ids[i]
            query_geometry_embedding, query_cad_embedding = embedding_loader.load_embeddings([query_cad_id], geometry_to_cad=geometry_to_cad)


            try:
                index = space_cad_id_dict[query_cad_id]
                ranking, confidence = do_experiment(geometry_space_embeddings=geometry_embeddings,
                                                    cad_space_embeddings=cad_embeddings, query_index=index,
                                                    space_cad_ids=space_cad_ids, test_name="exp" + str(i),
                                                    visualize=visualize,
                                                    geometry_loader=geometry_loader,
                                                    output_folder=output_folder,
                                                    encoder_type=encoder_type, save_3d=vis_3d)
            except:
                adjusted_geometry_embeddings = torch.concatenate([query_geometry_embedding, geometry_embeddings], dim=0)
                adjusted_cad_embeddings = torch.concatenate([query_cad_embedding, cad_embeddings], dim=0)
                adjusted_space_cad_ids = [query_cad_id] + space_cad_ids

                ranking, confidence = do_experiment(geometry_space_embeddings=adjusted_geometry_embeddings,
                                                    cad_space_embeddings=adjusted_cad_embeddings, query_index=0,
                                                    space_cad_ids=adjusted_space_cad_ids, test_name="exp" + str(i),
                                                    visualize=visualize,
                                                    geometry_loader=geometry_loader,
                                                    output_folder=output_folder,
                                                    encoder_type=encoder_type, save_3d=vis_3d)

            total_ranking += ranking
            total_confidence += confidence
            rank_holder.add_rank(ranking)
            rank_holder_per.add_rank(ranking)
    else:
        num_exp = num_geometries

        for i in range(num_exp):
            if i < plot_first_n_retrievals:
                visualize = True
                Path(output_folder).mkdir(parents=True, exist_ok=True)
            else:
                visualize = False
            ranking, confidence = do_experiment(geometry_space_embeddings=geometry_embeddings,
                                                cad_space_embeddings=cad_embeddings, query_index=i,
                                                space_cad_ids=space_cad_ids, test_name="exp" + str(i),
                                                visualize=visualize,
                                                geometry_loader=geometry_loader,
                                                output_folder=output_folder,
                                                encoder_type=encoder_type)

            total_ranking += ranking
            total_confidence += confidence
            rank_holder.add_rank(ranking)
            rank_holder_per.add_rank(ranking)

    if print_stats:
        print("====")
        print("Encoder:", contrastive_model_name, "ckpt-" + str(ckpt))
        print("Retrieval from", num_geometries, "geometries/cad")
        print("-")
        rank_holder_per.print_scores(num_exp)
        print("-")
        rank_holder.print_scores(num_exp)

        print("avg_ranking", total_ranking / num_exp, "avg_confidence", total_confidence / num_exp)
    return rank_holder, rank_holder_per

def normal_evaluation(checkpoint, encoder_type, contrastive_model_name):
    # aggregate all the rank holders together
    ranking_categories = [1]

    evaluate_on_augmented = False

    evaluate_retrieval(ckpt=checkpoint, encoder_type=encoder_type, contrastive_model_name=contrastive_model_name, evaluate_on_augmented=evaluate_on_augmented,
                       ranking_categories=ranking_categories, num_geometries=2048, plot_first_n_retrievals=5, print_stats=True, subfolder_name="Synthbal/", randomize_subbatch=True)



def plot_model_across_retrieval_batch_sizes(checkpoint, encoder_type, contrastive_model_name):
    # Special case options ----------------
    normalize = False

    evaluate_on_augmented = False # Default False

    min_seq_len = None # Default None
    max_seq_len = None # Default None

    num_bootstrap_repeats = 100  # Default 50

    balance_batches = False     # Default False
    batch_sizes = None          # Default None
    # if balance_batches:
    # batch_sizes = [10, 128, 512]
    # End special cases -------------------

    eval_model_across_retrieval_batch_sizes(checkpoint, encoder_type, contrastive_model_name, num_bootstrap_repeats=num_bootstrap_repeats,
                                            evaluate_on_augmented=evaluate_on_augmented, batch_sizes=batch_sizes,
                                            min_seq_len=min_seq_len, max_seq_len=max_seq_len, balanced_batch=balance_batches, do_plot=False, normalize=normalize, geom_subdir="meshes/")

def eval_model_across_retrieval_batch_sizes(checkpoint, encoder_type, results_path, num_bootstrap_repeats,
                                            requested_cad_ids=None, sample_n_from_requested_cad_ids=None, do_plot=True,
                                            batch_sizes=None, evaluate_on_augmented=False, min_seq_len=None, max_seq_len=None,
                                            balanced_batch=False, normalize=False, geom_subdir=None):

    if batch_sizes is None:
        batch_sizes = [10, 128, 1024, 2048]

    # aggregate all the rank holders together
    ranking_categories = [1, 2, 5]

    absolute_ranks = util.DictList()

    assert not (balanced_batch and (min_seq_len is not None or max_seq_len is not None))

    for batch_size in batch_sizes:
        print("Batch size", batch_size)
        geometry_to_cad, geometry_loader, embedding_loader = create_classes(checkpoint, encoder_type, results_path,
                                                                            batch_size, use_augmented=evaluate_on_augmented,
                                                                            min_seq_len=min_seq_len, max_seq_len=max_seq_len, normalize=normalize, geom_subdir=geom_subdir)

        if not balanced_batch:
            batch_loader = batch_creator.RandomBatchCreator(geometry_loader=geometry_loader, encoder_type=encoder_type)
            # batch_loader = None
        else:
            batch_loader = batch_creator.BalancedBatchCreator(geometry_loader=geometry_loader, num_bins=4, encoder_type=encoder_type)

        for _ in tqdm(range(num_bootstrap_repeats)):
            requested_cad_ids_batch = None
            if requested_cad_ids is not None:
                if sample_n_from_requested_cad_ids is not None:
                    indices = np.random.choice(np.arange(len(requested_cad_ids)), size=sample_n_from_requested_cad_ids, replace=True)
                    requested_cad_ids_batch = [requested_cad_ids[ind] for ind in indices]
                else:
                    indices = np.random.choice(np.arange(len(requested_cad_ids)), size=batch_size, replace=True)
                    requested_cad_ids_batch = [requested_cad_ids[ind] for ind in indices]

            if encoder_type == "random":
                rank_holder, rank_holder_per = evaluate_random_retrieval(ranking_categories=ranking_categories, num_geometries=batch_size, print_stats=False)
            else:
                rank_holder, rank_holder_per = evaluate_retrieval(ckpt=checkpoint, encoder_type=encoder_type,
                                                                  contrastive_model_name=results_path,
                                                                  ranking_categories=ranking_categories,
                                                                  num_geometries=batch_size,
                                                                  randomize_subbatch=True,
                                                                  print_stats=False,
                                                                  requested_cad_ids=requested_cad_ids_batch,
                                                                  geometry_to_cad=geometry_to_cad,
                                                                  geometry_loader=geometry_loader,
                                                                  embedding_loader=embedding_loader,
                                                                  evaluate_on_augmented=evaluate_on_augmented,
                                                                  batch_loader=batch_loader)

            absolute_ranks.add_to_key(batch_size, rank_holder)

    num_tests = len(batch_sizes)
    num_rankings = len(ranking_categories)

    # Now plot
    if do_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        width = 0.2
        x_range = np.arange(num_tests) * (num_rankings + 1) * width
        # x_range = np.arange(num_tests, step=(num_rankings + 1) * width)

        model_names = [str(batch_size) for batch_size in batch_sizes]
        model_label_x_offset = width * (num_rankings / 2.0) - width / 2

        ax.grid(True, zorder=0)
        ax.grid(True, which='minor', color=(0.9, 0.9, 0.9), linestyle='--', zorder=0)
        bar_x_offset = 0

    name = results_path + "_ckpt-" + str(checkpoint)
    print(name)
    model_values = {}
    model_errors = {}
    for rank in ranking_categories:
        print("Rank", rank)
        model_values_for_rank = {}
        model_errs_for_rank = {}
        for batch_size in batch_sizes:
            # There are multiple experiments: for each batch size, gether all numbers and get mean/std
            batch_values = []
            for rank_holder_abs in absolute_ranks.get_key(batch_size):
                number = rank_holder_abs.rank_holder[rank]
                if requested_cad_ids is None:
                    percent = number * 100.0 / rank_holder_abs.num_in_space
                elif sample_n_from_requested_cad_ids is None:
                    percent = number * 100.0 / len(requested_cad_ids)
                else: # sample_n is not none
                    percent = number * 100.0 / sample_n_from_requested_cad_ids # TODO note that this is not accurate if this value > len(requested_cad_ids)
                batch_values.append(percent)

            # now get statistics
            model_values_for_rank[batch_size] = np.mean(batch_values)
            model_errs_for_rank[batch_size] = np.std(batch_values)

            print("Batch:", batch_size, "\tMean", np.mean(batch_values))#, "\tstd", np.std(batch_values))

        model_values[rank] = model_values_for_rank
        model_errors[rank] = model_errs_for_rank
        if do_plot:
            ax.bar(x_range + bar_x_offset, list(model_values_for_rank.values()), width, label=str(rank), zorder=10)
            plt.errorbar(x_range + bar_x_offset, list(model_values_for_rank.values()), list(model_errs_for_rank.values()),
                     fmt ='none', capsize=5, capthick=1, ecolor='black', zorder=12)
            bar_x_offset += width

    if do_plot:
        ax.set_xticks(x_range + model_label_x_offset, model_names)
        ax.set_title('Top-n Accuracy\n' + name)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Retrieval Space Size")

        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', bottom=False)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # ax.set_aspect('equal', adjustable='box')
        plt.show()

    return {"mean": model_values, "std": model_errors}



def specific_retrieval_images(checkpoint, encoder_type, contrastive_model_name):
    vis_3d = True

    for complexity in list(cad_visualization_sets.keys()):
        subfolder_name = complexity
        visualization_set = cad_visualization_sets[subfolder_name]

        ranking_categories = [1, 2, 5]

        evaluate_retrieval(ckpt=checkpoint, encoder_type=encoder_type, contrastive_model_name=contrastive_model_name,
                           ranking_categories=ranking_categories, num_geometries=2048, plot_first_n_retrievals=len(visualization_set),
                           print_stats=False, requested_cad_ids=visualization_set, vis_3d=vis_3d, subfolder_name=subfolder_name)


import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Eval Pipeline")
    parser.add_argument("-encoder_type", "--encoder_type", type=str, required=False, default=None,
                        help="pc, pcn, mesh_feast")
    parser.add_argument("-contrastive_model_name", "--contrastive_model_name", type=str, required=False, default=None,
                        help="name of the contrastive model")
    parser.add_argument("-checkpoint", "--checkpoint", type=str, required=False, default=None,
                        help="checkpoint of retrieval model")
    parser.add_argument("-eval_method", "--eval_method", type=str, required=False, default="accuracy", choices=["accuracy", "images"],
                        help="Method of evaluation")
    args = parser.parse_args()


    # Default Choices =============
    # normal_evaluation()
    if args.eval_method == "accuracy":
        plot_model_across_retrieval_batch_sizes(args.checkpoint, args.encoder_type, args.contrastive_model_name)

    # Visualize ==========
    elif args.eval_method == "images":
        specific_retrieval_images(args.checkpoint, args.encoder_type, args.contrastive_model_name)



