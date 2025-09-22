import pyvista as pv
import trimesh
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import paths
from geometry.geometry_data import GeometryLoader
from geometry import trimesh_util
from geometry import pyvista_util
from pathlib import Path
from utils.util import DictList
from utils.visualizer_util import DaVinciVisualizer
from utils import histogram_visualizer
import matplotlib.ticker as ticker


def show_full_dataset(data_root, mesh_folder):
    geometry_loader1 = GeometryLoader(data_root, "test", with_normals=True, geometry_subdir=mesh_folder)

    visualizer = DaVinciVisualizer(geometry_loader1)
    use_case = "default"

    data_ids = geometry_loader1.get_cad_ids_of_sequence_length(0, 10, cad=True, mesh=True, cloud=True)
    # data_ids = geometry_loader1.get_all_valid_data(cad=True, mesh=True)

    for data_id in tqdm(data_ids[0:40]):
        print(data_id)
        # cad_program = geometry_loader1.load_cad(data_id)
        # print(cad_program)
        # now pplot
        default_size = 800
        pl = pv.Plotter(shape=(1, 4), window_size=[default_size * 3, default_size * 1])
        pl.subplot(0, 0)
        visualizer.plot_picture(data_id=data_id, style="cloud", pl=pl, text="Cloud", use_case=use_case)

        pl.subplot(0, 1)
        visualizer.plot_picture(data_id=data_id, style="mesh", pl=pl, text="mesh", use_case=use_case)

        pl.subplot(0, 2)
        visualizer.plot_picture(data_id=data_id, style="cad", pl=pl, text="CAD", use_case=use_case)

        pl.subplot(0, 3)
        visualizer.plot_picture(data_id=data_id, style="pcn", pl=pl, text="PCN", use_case=use_case, normals=True)

        visualizer.plot_picture(data_id=data_id, style="cad", text="CAD", save_path="temp.png")

        pl.link_views()
        pl.show()

def show_dataset_normals(data_root, mesh_folder):
    geometry_loader1 = GeometryLoader(data_root, "test", with_normals=True, geometry_subdir=mesh_folder)

    visualizer = DaVinciVisualizer(geometry_loader1)
    use_case = "default"

    # data_ids = geometry_loader1.get_cad_ids_of_sequence_length(40, 80)

    for data_id in tqdm(geometry_loader1.get_all_valid_data(cad=True, mesh=True)[:10]):
        verts = geometry_loader1.load_cloud(data_id, 2048)
        _, estimated_normals = geometry_loader1.get_cloud_normals_estimated(verts[..., :3])
        mesh_verts, _, faces = geometry_loader1.load_mesh_vef(data_id)

        # now pplot
        default_size = 800
        pl = pv.Plotter(shape=(1, 3), window_size=[default_size * 3, default_size * 1])
        pl.subplot(0, 0)
        visualizer.plot_picture(verts=verts[..., :3], style="cloud", pl=pl, text="mesh normals", use_case=use_case, normals=verts[..., 3:])

        pl.subplot(0, 1)
        visualizer.plot_picture(verts=verts[..., :3], style="cloud", pl=pl, text="estimated normals", use_case=use_case, normals=estimated_normals)

        pl.subplot(0, 2)
        visualizer.plot_picture(verts=mesh_verts[..., :3], faces=faces, style="mesh", pl=pl, text="vertex normals", use_case=use_case, normals=mesh_verts[..., 3:])


        pl.link_views()
        pl.show()

def show_geometries_in_grid(data_root, mesh_folder, r=4, c=4):
    geometry_loader1 = GeometryLoader(data_root, "test", with_normals=False, geometry_subdir=mesh_folder)
    visualizer = DaVinciVisualizer(geometry_loader1)

    # data_ids = geometry_loader1.get_all_valid_data(cad=True, mesh=True)
    data_ids = geometry_loader1.get_cad_ids_of_sequence_length(3, 10, mesh=True)
    print(len(data_ids))

    num_data = len(data_ids)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data/ num_visualized))
    default_size = 150

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c), window_size=[default_size * c, default_size * r])
        for ri in range(r):
            for ci in range(c):
                idx = i * r * c + ri * c + ci

                if idx >= num_data:
                    break
                pl.subplot(ri, ci)

                data_id = data_ids[idx]

                # visualizer.plot_picture(pl=pl, data_id=data_id, style="cad", use_case="default", text="",
                                        # override_color=visualizer.default_scan_color,
                                        # center=True)

                visualizer.plot_picture(pl=pl, data_id=data_id, style="mesh", use_case="simple", text=str(data_id), center=True)
        pl.link_views()
        pl.show()

def show_stls_in_grid(data_root, stl_dir, r=4, c=4):
    # Use case: Examine STLs and make sure they are fine

    geometry_loader1 = GeometryLoader(data_root, phase=None, with_normals=False, stl_directory=stl_dir)
    visualizer = DaVinciVisualizer(geometry_loader1)

    data_ids = geometry_loader1.get_all_valid_data(stl=True)
    # data_ids = geometry_loader1.get_cad_ids_of_sequence_length(20, 35)
    print(len(data_ids))

    num_data = len(data_ids)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data/ num_visualized))
    default_size = 150

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c), window_size=[default_size * c, default_size * r])
        for ri in range(r):
            for ci in range(c):
                idx = i * r * c + ri * c + ci

                if idx >= num_data:
                    break
                pl.subplot(ri, ci)

                data_id = data_ids[idx]

                visualizer.plot_picture(pl=pl, data_id=data_id, style="stl", use_case="simple", text=str(data_id))
        pl.link_views()
        pl.show()


def show_cloud_mesh_cad(data_root, mesh_folder):
    geometry_loader1 = GeometryLoader(data_root, "train", with_normals=False, geometry_subdir=mesh_folder)
    visualizer = DaVinciVisualizer(geometry_loader1)
    use_case = "default"
    for data_id in tqdm(geometry_loader1.get_all_valid_data(cad=True, mesh=True)[:10]):
        # now pplot
        default_size = 800
        pl = pv.Plotter(shape=(1, 3), window_size=[default_size * 3, default_size * 1])
        pl.subplot(0, 0)
        visualizer.plot_picture(data_id=data_id, style="cloud", pl=pl, text="Cloud", use_case=use_case)

        pl.subplot(0, 1)
        visualizer.plot_picture(data_id=data_id, style="mesh", pl=pl, text="mesh", use_case=use_case)

        pl.subplot(0, 2)
        visualizer.plot_picture(data_id=data_id, style="cad", pl=pl, text="CAD", use_case=use_case)

        visualizer.plot_picture(data_id=data_id, style="cad", text="CAD", save_path="temp.png")

        pl.link_views()
        pl.show()


def compare_cloud_mesh(data_root, save_dir):
    mesh_folder = "meshes/"

    geometry_loader1 = GeometryLoader(data_root, "train", with_normals=False, geometry_subdir="meshes_original/")
    geometry_loader2 = GeometryLoader(data_root, "train", with_normals=False, geometry_subdir="meshes_rm2kfix/")
    geometry_loader2kf = GeometryLoader(data_root, "train", with_normals=False, geometry_subdir="meshes_rm2kthinfix/")
    geometry_loader3 = GeometryLoader(data_root, "train", with_normals=False, geometry_subdir="meshes/")

    for data_id in tqdm(geometry_loader1.get_all_valid_data(cad=True, mesh=True)[:10]):
        try:
            cloud = geometry_loader1.load_cloud(data_id, as_tensor=False)
            verts1, _, faces1 = geometry_loader1.load_mesh_vef(data_id, as_tensor=False)

            verts2, _, faces2 = geometry_loader2.load_mesh_vef(data_id, as_tensor=False)

            verts2kf, _, faces2kf = geometry_loader2kf.load_mesh_vef(data_id, as_tensor=False)

            verts3, _, faces3 = geometry_loader3.load_mesh_vef(data_id, as_tensor=False)
        except:
            print("Load error")
            continue

        # now pplot
        pyvista_util.show_geometries_in_grid([(cloud,), None,
                                              (verts1, faces1), (verts1,),
                                              (verts2, faces2), (verts2,),
                                              (verts2kf, faces2kf), (verts2kf,),
                                              (verts3, faces3), (verts3,)],
                                             names=["cloud", None,
                                                    "mesh_orig", "mesh_orig verts",
                                                    "mesh_rm2k", "mesh_rm2k verts",
                                                    "mesh_rm2kfix", "mesh_rm2kfix verts",
                                                    "mesh_rm10k", "mesh_rm10k verts"],
                                             r=5, c=2, transpose=True,
                                             save_name = save_dir + "mesh_" + str(data_id[5:]))
        # print(data_id)
        # print("num verts", len(verts))
        # print("num cloud", len(cloud))
        # print("---")

def collect_pc_data(data_root):
    geometry_loader = GeometryLoader(data_root, "train")
    for data_id in tqdm(geometry_loader.all_data[:3000]):
        try:
            cloud = geometry_loader.load_cloud(data_id, as_tensor=False)
        except:
            print("Load error")
            continue

        bounds_min = np.min(cloud, axis=0)
        bounds_max = np.max(cloud, axis=0)
        bounds_length = bounds_max - bounds_min

        print(bounds_length)


def collect_data(data_root, mesh_folder, verts=True, edges=True, faces=True, cloud=True):
    # Check if cloud exists
    # Check if model loads correctly
    edge_list = []
    vert_list = []
    face_list = []
    bound_list_bad = []
    bound_list_working = []

    geometry_loader = GeometryLoader(data_root, "train", geometry_subdir=mesh_folder)
    for data_id in tqdm(geometry_loader.get_all_valid_data(mesh=True)[:3000]):
        try:
            verts, edges, faces = geometry_loader.load_mesh_vef(data_id, as_tensor=False)
        except:
            print("Load error")
            continue

        if np.max(faces) > len(verts):
            print("face index error", data_id)
            continue
        if np.max(edges) > len(verts):
            print("edge index error", data_id)
            continue

        vert_list.append(len(verts))
        edge_list.append(len(edges))
        face_list.append(len(faces))


        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # neighbors = geometry_loader.mesh_neighbors(edges, flattened=True)
        mesh_aux = trimesh_util.MeshAuxilliaryInfo(mesh)

        max_bound = max(mesh_aux.bound_length)
        bound_ratios = mesh_aux.bound_length / max_bound
        bound_ratios = np.sort(bound_ratios)
        bound_ratios = bound_ratios[:2] # only grab the smaller 2

        if len(verts) < 2000:
            # trimesh_util.show_mesh(mesh)
            # print(data_id)
            bound_list_bad.append(bound_ratios)
        else:
            bound_list_working.append(bound_ratios)


    return {
        "num_verts": vert_list,
        "num_edges": edge_list,
                "num_faces": face_list,
                "bound_list_bad": np.array(bound_list_bad),
                "bound_list_working": np.array(bound_list_working)
            }

def analysis(data_root, mesh_folder):
    save_path = "visualization/dataset/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    # compare_cloud_mesh(data_root, save_path)
    # collect_pc_data(data_root)
    out = collect_data(data_root, mesh_folder)
    vert_list = out["num_verts"]
    vert_list = np.array(vert_list)

    in_bounds = (vert_list >= 2040) & (vert_list <= 2500)
    num_in_bound = len(vert_list[in_bounds])
    print(num_in_bound)
    #
    sns.histplot(data=vert_list, log_scale=False)
    plt.show()

def analysis_extrudes_curves(data_root):
    # checks the max number of extrudes / cad vec and
    # max number of commands / extrude
    from cadlib.util import get_extrude_idx
    num_extrudes = DictList()
    num_commands = DictList()
    geometry_loader = GeometryLoader(data_root, None)
    pbar = tqdm(geometry_loader.get_all_valid_data(cad=True))
    for data_id in pbar:
        cmd, args = geometry_loader.load_cad(data_id, tensor=False, as_single_vec=False)
        extrude_idxs = get_extrude_idx(cmd)
        num_extrudes.add_to_key(len(extrude_idxs), 1)
        lengths = []
        for i in range(len(extrude_idxs)):
            if i == 0:
                num_commands.add_to_key(extrude_idxs[i], 1)
                lengths.append(extrude_idxs[0])
            else:
                num_commands.add_to_key(extrude_idxs[i] - extrude_idxs[i-1], 1)
                lengths.append(extrude_idxs[i] - extrude_idxs[i-1])
        pbar.set_postfix({"max c/e": max(num_commands.keys())})
        if max(lengths) > 30:
            print(extrude_idxs, data_id, cmd)

    for key, val in num_extrudes.items():
        num_extrudes.dictionary[key] = len(val)
    for key, val in num_commands.items():
        num_commands.dictionary[key] = len(val)

    print("max extrudes:", max(num_extrudes.keys()))
    print("max commands/extrude:", max(num_commands.keys()))
        



def show_poor_mesh_ratios(data_root):
    save_path = "visualization/dataset/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    # compare_cloud_mesh(data_root, save_path)
    # collect_pc_data(data_root)
    out = collect_data(data_root)
    bound_list_bad = out["bound_list_bad"]
    bound_list_working = out["bound_list_working"]
    #
    plt.scatter(bound_list_working[:, 0], bound_list_working[:, 1], alpha=0.2)
    plt.scatter(bound_list_bad[:, 0], bound_list_bad[:, 1], alpha=0.2)

    plt.show()

def show_cad_sequence_synthetic(data_root):
    geometry_loader1 = GeometryLoader(data_root, "test", with_normals=False)
    geometry_loader1.get_cad_ids_of_sequence_length(cad=True, mesh=False, cloud=False)
    sequence_lengths = geometry_loader1.sequence_length_cache

    data_ids = list(sequence_lengths.keys())
    seq_lengths_orig = [sequence_lengths[key] for key in sequence_lengths if key.find("_") == -1]
    seq_lengths_synthetic = [sequence_lengths[key] for key in sequence_lengths if key.find("_") != -1]

    print(len(data_ids))
    print(len(seq_lengths_orig))
    print(len((seq_lengths_synthetic)))


    fig, ax = plt.subplots()
    ax.hist([list(seq_lengths_orig), list(seq_lengths_synthetic)], stacked=True, bins=56, label=["original", "synthetic"], density=True)
    # ax.hist(list(sequence_lengths.values()), bins=57)
    ax.set_yscale('log')
    ax.set_xlabel("Sequence Lengths")
    # ax.grid(axis='y')
    ax.grid(axis='x', visible=True, which='minor', linestyle='--')
    ax.grid(axis='x', visible=True, which='major', linestyle='--')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # ax.set_ylim(ymin=1)
    ax.set_ylim(ymax=0.5)
    ax.set_xlim(xmin=min(sequence_lengths.values()))
    ax.set_xlim(xmax=max(sequence_lengths.values()))
    # ax.legend()
    plt.show()

def get_statistics(data_root):
    from cadlib.macro import LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX
    geometry_loader = GeometryLoader(data_root, "test", with_normals=False, splits_file="filtered_data")

    # histogram of sequence lengths
    geometry_loader.get_cad_ids_of_sequence_length(cad=True, mesh=False, cloud=False)
    sequence_lengths = geometry_loader.sequence_length_cache

    # print(len(geometry_loader.get_cad_ids_of_sequence_length(min_length=6, max_length=6, cad=True)))
    # print(len(geometry_loader.get_cad_ids_of_sequence_length(min_length=5, max_length=5, cad=True)))
    # print(len(geometry_loader.get_cad_ids_of_sequence_length(min_length=4, max_length=4, cad=True)))
    # print(len(geometry_loader.get_cad_ids_of_sequence_length(min_length=None, max_length=6, cad=True)))

    # print(len(sequence_lengths[6]))

    # histogram of extrude lengths
    # ratios of arcs/circles/lines
    # ratio of extrude booleans (union, new, subtract, intersect)
    # ratio of extrude end conditions (blind, both, symm)

    class StatisticsAggregator:
        def __init__(self):
            self.sketch_type_dictlist = {
                "arc": 0,
                "line": 0,
                "circle": 0
            }
            self.sketch_type_ratios = []
            self.extrude_bool_dictlist = {
                "new": 0,
                "add": 0,
                "subtract": 0,
                "intersect": 0
            }
            self.extrude_end_dictlist = {
                "one": 0,
                "two": 0,
                "symmetric": 0,
            }
            self.sketch_lengths = []
            self.extrude_lengths = []

            self.args = {
                "arc": [],
                "line": [],
                "circle": [],
                "extrude": [],
            }

        def add_cad(self, command, args):
            ext_indices = np.where(command == EXT_IDX)[0]
            cad_vec = np.hstack([command[:, np.newaxis], args])
            split_cad_vec = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

            # Find the arc/circle/line commands
            num_line = np.sum(command == LINE_IDX)
            num_circle = np.sum(command == CIRCLE_IDX)
            num_arc = np.sum(command == ARC_IDX)
            total_commands = num_line + num_circle + num_arc
            self.sketch_type_dictlist["line"] += num_line
            self.sketch_type_dictlist["arc"] += num_arc
            self.sketch_type_dictlist["circle"] += num_circle
            self.sketch_type_ratios.append(np.array([num_arc, num_line, num_circle]) / total_commands)

            # get the extrude parameters
            self.extrude_bool_dictlist["new"] += np.sum(args[:, 15] == 0)
            self.extrude_bool_dictlist["add"] += np.sum(args[:, 15] == 1)
            self.extrude_bool_dictlist["subtract"] += np.sum(args[:, 15] == 2)
            self.extrude_bool_dictlist["intersect"] += np.sum(args[:, 15] == 3)

            self.extrude_end_dictlist["one"] += np.sum(args[:, 15] == 0)
            self.extrude_end_dictlist["two"] += np.sum(args[:, 15] == 2)
            self.extrude_end_dictlist["symmetric"] += np.sum(args[:, 15] == 1)

            self.args["arc"] += list(args[command == ARC_IDX][:, [0, 1, 2, 3]])
            self.args["line"] += list(args[command == LINE_IDX][:, [0, 1]])
            self.args["circle"] += list(args[command == CIRCLE_IDX][:, [0, 1, 4]])
            self.args["extrude"] += list(args[command == EXT_IDX][:, 5:])

            # sketch lengths
            for split in split_cad_vec:
                self.sketch_lengths.append(len(split) - 2)

            # find the extrude params
            self.extrude_lengths.append(len(split_cad_vec))

        def get_arg_limits(self):
            limits = {}
            for key in self.args.keys():
                values = self.args[key]
                values = np.array(values)
                minim = np.min(values, axis=0)
                maxim = np.max(values, axis=0)
                limits[key] = np.vstack([maxim, minim])

            return limits

    statistics = StatisticsAggregator()
    statistics_per_complexity = {}
    for i in range(3, 60):
        statistics_per_complexity[i] = StatisticsAggregator()

    print("Analyzing data")
    for data_id in tqdm(geometry_loader.get_all_valid_data(cad=True), smoothing=0.1):
        command, args = geometry_loader.load_cad(data_id=data_id, tensor=False, pad=False)
        length = sequence_lengths[data_id]

        statistics.add_cad(command, args)
        statistics_per_complexity[length].add_cad(command, args)

    def print_dict_values(dictionary):
        dict_values = np.array(list(dictionary.values()))

        print(list(dictionary.keys()), "(", list(np.round(dict_values / np.sum(dict_values), 3)), ")")


    ## Lengths

    fig, axs = plt.subplots(3)
    ax = axs[0]
    bins = max(sequence_lengths.values()) - min(sequence_lengths.values())
    print(bins)
    ax.hist(sequence_lengths.values(),  bins=bins, density=True)
    ax.set_yscale('log')
    ax.set_xlabel("Sequence Lengths")
    ax.set_ylim(bottom=5e-6)
    # ax.grid(axis='y')

    ax.grid(axis='x', visible=True, which='minor', linestyle='--')
    ax.grid(axis='x', visible=True, which='major', linestyle='--')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # ax.set_ylim(ymin=1)
    ax.set_ylim(ymax=0.5)

    ax.set_xlim(xmin=min(sequence_lengths.values()))
    ax.set_xlim(xmax=max(sequence_lengths.values()))


    ax = axs[1]
    bins = max(statistics.sketch_lengths) - min(statistics.sketch_lengths) + 1
    ax.hist(statistics.sketch_lengths,  bins=bins, density=True)
    ax.set_xlabel("Sketch Lengths")
    ax.set_yscale('log')
    ax.grid(axis='y')
    ax.set_ylim(ymin=1)

    ax = axs[2]
    bins = max(statistics.extrude_lengths) - min(statistics.extrude_lengths) + 1
    ax.hist(statistics.extrude_lengths,  bins=bins, density=True)
    ax.set_xlabel("Extrude Lengths")
    ax.set_yscale('log')
    ax.grid(axis='y')
    ax.set_ylim(ymin=1)

    plt.show()


    ## Parameter distribution
    print_dict_values(statistics.sketch_type_dictlist)
    print_dict_values(statistics.extrude_end_dictlist)
    print_dict_values(statistics.extrude_bool_dictlist)
    print(statistics.get_arg_limits())

    for key in statistics.args.keys():
        data = np.array(statistics.args[key])
        dim = data.shape[1]


        row = int(np.sqrt(dim))
        col = int(np.ceil(dim / row))

        # fig, axs = plt.subplots(row, col)
        fig = plt.figure()
        fig.suptitle(key)
        for i in range(dim):
            ax = fig.add_subplot(row, col, i+1)
            # r = i // row
            # c = i % col
            # ax = axs[r][c]
            ax.hist(data[:, i], bins=20)
            ax.set_xlabel(i)
            ax.grid(axis='y')

    plt.show()

    colormap = plt.get_cmap("gist_rainbow")
    length_colors = colormap(np.array(list(range(0, 60)))/60.0)

    ## Parameter distribution per length
    for key in statistics.args.keys():
        data = np.array(statistics.args[key])
        dim = data.shape[1]
        fig = plt.figure()
        fig.suptitle(key)
        for i in range(dim):
            data = []
            colors = []
            for length in range(3, 60):
                seq_data = np.array(statistics_per_complexity[length].args[key])
                if len(seq_data) > 0:
                    data.append(seq_data[:, i])
                    colors.append(length_colors[length])
            multihist = histogram_visualizer.MultiHistogram(data=data, bins=25, num_figs=dim, colors=colors)
            ax = multihist.plot(fig, idx=i+1)
            ax.set_xlabel(i)

    plt.show()

    fig, ax = plt.subplots()
    plotter = histogram_visualizer.VectorPlotter(np.array(statistics.sketch_type_ratios), ["arc", "line", "circle"])
    plotter.plot(ax)
    plt.show()


if __name__=="__main__":
    data_root = "/mnt/barracuda/CachedDatasets/GenCAD3D/"
    # data_root = "/mnt/barracuda/CachedDatasets/DaVinci_CAD_Augmented_NA/"
    mesh_folder = "meshes_rm2kfix/"

    # data_root = paths.HOME_PATH + "scans/50_scans/"
    # mesh_folder = "meshes_rmquad/"

    data_root = paths.DATA_PATH + "GenCAD3D/"

    # analysis_extrudes_curves(data_root)

    get_statistics(data_root)
    # show_cad_sequence_synthetic(data_root)

    # show_full_dataset(data_root, mesh_folder)
    # show_dataset_normals(data_root, mesh_folder)

    # Grid Plots =======================
    show_geometries_in_grid(data_root, mesh_folder=mesh_folder, r=6, c=9)
    # show_stls_in_grid(data_root, stl_dir="remeshed_stls/", r=4, c=4)

    # compare_cloud_mesh(data_root=data_root, save_dir="temp.png")
    # show_cloud_mesh_cad(data_root, mesh_folder)
    # analysis(data_root, mesh_folder)