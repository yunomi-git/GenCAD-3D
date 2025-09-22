import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
from time import time

import paths
import utils.visualizer_util as visualizer_util

import numpy as np
from pathlib import Path
import contrastive.contrastive_evaluation_util as mesh_evaluation_util
from contrastive.contrastive_evaluation_util import GeometryEmbeddingSpace
from geometry.geometry_data import GeometryLoader
import umap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm


if __name__=="__main__":

    # Goal: compare latent space of geometries to cad
    # 1. get latent space of geometries
    # 2. get latent space of cats
    # 3. put together and reduce dimensionality
    # 4. plot them on latent space separately
    num_geometries = 512
    generation_embeddings = False
    num_pictures = 100
    output_folder = "visualization/latents_plot/" + "master" + str(num_geometries) + "/"
    Path(output_folder).mkdir(parents=True, exist_ok=True)


    data_root = paths.DATA_PATH + "GenCAD3D/"
    geometry_loader_master = GeometryLoader(data_root=data_root, phase="test",
                                     with_normals=True,
                                     geometry_subdir="meshes/",
                                     stl_directory="stls/")

    embedding_models = [
        {"encoder_type": "mesh_feast",
         "contrastive_model_name": "mesh_SynthBal_1M",
         "ckpt": "latest",
         "visual_style": "mesh",
         "visual_normals": False,
         },
        {"encoder_type": "pc",
         "results_path": "pc_SynthBal_1M",
         # "ckpt": latest,
         "visual_style": "cloud",
         "visual_normals": False,
         },
        {"encoder_type": "pc",
         "results_path": "pcn_SynthBal_1M",
         "ckpt": "latest",
         "visual_style": "pcn",
         "visual_normals": False,
         },
    ]

    geometry_embeddings_list = []
    cad_embeddings = None
    space_cad_ids = None
    for embedding_model in embedding_models:
        encoder_type = embedding_model["encoder_type"]
        results_path = embedding_model["results_path"]
        ckpt = embedding_model["ckpt"]

        geometry_to_cad = mesh_evaluation_util.GeometryToCAD(encoder_type=encoder_type,
                                                             contrastive_model_name="results/" + results_path + "/",
                                                             encoder_ckpt_num=ckpt)
        geometry_loader = GeometryLoader(data_root=data_root, phase="test",
                                         with_normals=geometry_to_cad.use_normals,
                                         geometry_subdir="meshes_rm2kfix/",
                                         stl_directory="stls/")

        cache_folder = "visualization/embedding_cache/"

        embedding_loader = GeometryEmbeddingSpace(encoder_type=encoder_type, num_geometries=num_geometries,
                                                  cache_parent_dir=cache_folder, geometry_loader=geometry_loader,
                                                  ckpt_name=results_path + str(ckpt), generation=generation_embeddings)
        embedding_loader.do_randomize_subbatch()

        geometry_embeds, _ = embedding_loader.load_geometry_space_embeddings(geometry_to_cad=geometry_to_cad)
        geometry_embeddings_list.append(geometry_embeds.cpu().detach().numpy())

        if cad_embeddings is None:
            cad_embeds, space_cad_ids = embedding_loader.load_cad_embeddings(geometry_to_cad=geometry_to_cad)
            cad_embeddings = cad_embeds.cpu().detach().numpy()

    # Load cad embedding




    all_embeddings = [cad_embeddings] + geometry_embeddings_list
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    # embeddings = np.concatenate([geometry_embeddings, cad_embeddings], axis=0)

    # Visualizer
    visualizer = visualizer_util.DaVinciVisualizer(geometry_loader=geometry_loader)

    ## FITTING ==================================================
    print("Fitting")

    fit_method = "tsne"
    if fit_method == "umap":
        n_components = 2
        n_neighbors = 15
        min_dist = 0.1
        metric = 'cosine'
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
    elif fit_method == "tsne":
        fit = TSNE(n_components=2, random_state=42, perplexity=40, learning_rate=400)
    else:
        fit = PCA(n_components=2)

    start = time()
    u = fit.fit_transform(all_embeddings)

    print("fitting time", time() - start)

    # Normalize u
    def normalize_to_range(x, scale):
        min_x = np.min(x)
        max_x = np.max(x)
        mean_x = np.mean(x)

        return (x - mean_x) / (max_x - min_x) * (2 * scale)

    u[:, 0] = normalize_to_range(u[:, 0], 10)
    u[:, 1] = normalize_to_range(u[:, 1], 10)
    rotate = True
    if rotate:
        temp = u[:, 0].copy()
        u[:, 0] = u[:, 1].copy()
        u[:, 1] = temp

    # Now extract the individual reductions
    u_geometries_list = []
    u_cad = None

    for i in range(len(embedding_models)):
        u_geometries_list.append(u[num_geometries * (i + 1):num_geometries * (i+2)])
    u_cad = u[:num_geometries]

    # PLOTTING =================================================================
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    app = pg.mkQApp("Scatter Plot Item Example")
    mw = QtWidgets.QMainWindow()
    mw.resize(800, 800)
    view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
    mw.setCentralWidget(view)
    mw.show()
    mw.setWindowTitle('pyqtgraph example: ScatterPlot')

    window = view.addPlot()
    window.setAspectLocked(True)



    # COLORING ===========================
    color_type = "model"
    cad_colors = []
    geometry_colors_list = []

    if color_type == "model":
        cad_colors = [visualizer.default_cad_color for _ in range(num_geometries)]
        geometry_colors_list = []
        for i_model in range(len(embedding_models)):
            style = embedding_models[i_model]["visual_style"]
            color = visualizer.default_colors[style]
            geometry_colors_list.append([color for _ in range(num_geometries)])

    if color_type == "correspondance":
        # Obtain custom color plot. Use cad as reference
        normalized_u_cad = np.zeros_like(u_cad)
        normalized_u_cad[:, 0] = normalize_to_range(u_cad[:, 0], 0.5) + 0.5
        normalized_u_cad[:, 1] = normalize_to_range(u_cad[:, 1], 0.5) + 0.5

        def get_color(x, y):
            return (int(x * 255), int(y * 255), 120, 255)

        cad_colors = [get_color(normalized_u_cad[i, 0], normalized_u_cad[i, 1]) for i in range(num_geometries)]
        geometry_colors_list = [cad_colors for _ in range(len(embedding_models))]
    elif color_type == "complexity":
        from cadlib.macro import (
            EOS_IDX
        )

        seq_lengths = []
        for data_id in space_cad_ids:
            command, args = geometry_loader.load_cad(data_id)
            seq_len = command.tolist().index(EOS_IDX)
            seq_lengths.append(seq_len)
        seq_lengths = np.array(seq_lengths)

        cad_colors = visualizer_util.label_to_color(num_classes=max(seq_lengths) - min(seq_lengths) + 1, classes=seq_lengths)
        cad_colors = list(cad_colors)
        geometry_colors_list = [cad_colors for _ in range(len(embedding_models))]
    elif color_type == "shapes":
        from cadlib.macro import (
            EOS_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX
        )

        shapes_ratios = []
        for data_id in space_cad_ids:
            command, args = geometry_loader.load_cad(data_id)
            command = command.tolist()
            seq_len = command.index(EOS_IDX)
            command = np.array(command)
            num_line = np.sum(command == LINE_IDX) / 4
            num_circle = np.sum(command == CIRCLE_IDX)
            num_arc = np.sum(command == ARC_IDX)
            num_commands = num_line + num_circle + num_arc

            shape_ratio = np.array([num_line, num_arc, num_circle]) / num_commands

            shapes_ratios.append(shape_ratio)

        shapes_ratios = np.array(shapes_ratios)

        cad_colors = shapes_ratios * 255
        cad_colors = np.concatenate([cad_colors, np.ones(len(space_cad_ids))[:, np.newaxis] * 255], axis=1)
        cad_colors = list(cad_colors)
        geometry_colors_list = [cad_colors for _ in range(len(embedding_models))]


    # ============================================
    # Colors created. Now plot!
    # Colors are ordered in [geometries, cad]
    scatter1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    spots = []
    # Geometries
    for i_model in range(len(embedding_models)):
        u_geometries = u_geometries_list[i_model]
        geometry_colors = geometry_colors_list[i_model]
        for i in range(num_geometries):
            spots.append({'pos': (u_geometries[i, 0], u_geometries[i, 1]),
                          'brush': pg.mkBrush(geometry_colors[i]),
                          # 'brush': pg.mkBrush(visualizer.default_mesh_color),
                          'data': {"data_id": space_cad_ids[i]}
                          })
    # CAD
    for i in range(num_geometries):
        spots.append({'pos': (u_cad[i, 0], u_cad[i, 1]),
                      'brush': pg.mkBrush(cad_colors[i]),
                      # 'brush': pg.mkBrush(visualizer.default_cad_color),
                      'data': {"data_id": space_cad_ids[i]}
                      })
    scatter1.addPoints(spots)
    window.addItem(scatter1)


    # PICTURES =====================================
    # Method
    # Given the list of x, y, data_id
    # Go through the list of data ids
    # plot the picture at the associated x and y
    save_dir = "temp/"
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    save_name = save_dir + "temp.png"
    def plot_image(data_id, x, y, style, normals=False):
        image_data = visualizer.plot_picture(data_id=data_id, style=style, save_path=save_name, use_case="icon", normals=normals)
        image_data = visualizer_util.remove_white(image_data)
        # flip upside down
        image_data = np.flipud(image_data)
        # load image
        img = pg.ImageItem(image=image_data, axisOrder='row-major')

        width = 1.2
        img.setRect(x - width/2, y-width/2, width, width)
        window.addItem(img)
    # Pictures
    for i_model in range(len(embedding_models)):
        u_geometries = u_geometries_list[i_model]
        style = embedding_models[i_model]["visual_style"]
        plot_normals = embedding_models[i_model]["visual_normals"]
        for i in tqdm(range(num_pictures)):
            x, y = u_geometries[i, 0], u_geometries[i, 1]
            data_id = space_cad_ids[i]
            if geometry_loader.check_valid_data(data_id=data_id, mesh=True, cad=True):
                if geometry_loader.calc_cad_sequence_length(data_id) > 10:
                    plot_image(data_id, x, y, style=style, normals=plot_normals)
    for i in tqdm(range(num_pictures)):
        x, y = u_cad[i, 0], u_cad[i, 1]
        data_id = space_cad_ids[i]
        if geometry_loader.calc_cad_sequence_length(data_id) > 10:
            plot_image(data_id, x, y, style="cad")

    pg.exec()
