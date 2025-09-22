import pyvista as pv
from geometry.geometry_data import GeometryLoader
from geometry.pyvista_util import convert_to_pv_mesh, get_normals_object_for_vis
import cv2
import numpy as np
from matplotlib import colormaps
from geometry import trimesh_util
import paths

class DaVinciVisualizer:
    def __init__(self, geometry_loader: GeometryLoader=None):
        self.geometry_loader = geometry_loader
        self.default_size = 800

        self.default_cloud_color = "#fe4a49"
        self.default_mesh_color = "#fed766"
        self.default_cad_color = "#a8e6cf"
        self.default_pcn_color = "#98ff98"
        self.default_pcn_normals_color = "#007848"
        self.default_brep_color = "#3366FF"

        self.default_scan_color = "#F67280"


        self.default_colors = {
            "cloud": self.default_cloud_color,
            "cad": self.default_cad_color,
            "mesh": self.default_mesh_color,
            "pcn": self.default_pcn_color,
            "scan": self.default_scan_color,
            "brep": self.default_brep_color
        }

    def set_default_pl_options(self, pl, text=None, use_case="default"):
        # pl.remove_all_lights()

        # if use_case == "default":
        #     pl.add_floor('-z', lighting=True,
        #                  color='white', pad=1.0, offset=0.01,
        #                  i_resolution=100, j_resolution=100)

        pl.enable_shadows()

        # if use_case != "simple":
        light = pv.Light(intensity=0.01,
                         position=(0, 0.2, 10.0),
                         shadow_attenuation=1.0,
                         focal_point=(0, 0, 0),)
        light.set_direction_angle(30, 120)
        pl.add_light(light)

        if text is not None:
            pl.add_text(text, color='black', font_size=24) #large font 24, 48

    def make_new_plot(self, headless=False, r=1, c=1):

        pl = pv.Plotter(shape=(r, c), window_size=[self.default_size*c, self.default_size*r], off_screen=headless)
        return pl
    
    def get_image_of_cad_shape(self, out_shape, save_path, text=None, pl=None):
        import os
        import trimesh 
        from OCC.Extend.DataExchange import write_stl_file

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

        self.plot_picture(verts=verts, faces=faces, save_path=save_path, style="cad", text=text, pl=pl)

        os.remove(temp_save_name)


    def plot_picture(self, style, data_id=None, verts=None, normals=None, faces=None,
                     pl=None, text=None, save_path=None, use_case="default", override_color=None,
                     center=False, save_path_3d=None):
        # Use cases: - place on existing plot, - just plot, -just save image, - just return image
        # normals: can manually pass in the normals
        if pl is None:
            pl = self.make_new_plot(headless=save_path is not None)
        if style == "cloud" or style == "pointcloud" or style == "pcn" or style == "pc":
            multiblock = self._add_cloud_to_plot(data_id=data_id, verts=verts, pl=pl, use_case=use_case, normals=style=="pcn", override_color=override_color, center=center)
        elif "mesh" in style:
            multiblock = self._add_mesh_to_plot(data_id=data_id, verts=verts, faces=faces, pl=pl, use_case=use_case, override_color=override_color, center=center)
        elif style == "cad":
            multiblock = self._add_cad_to_plot(data_id=data_id, verts=verts, faces=faces, pl=pl, use_case=use_case, override_color=override_color, center=center)
        elif style == "stl":
            multiblock = self._add_stl_to_plot(data_id=data_id, pl=pl, use_case=use_case, override_color=override_color, center=center)
        else:
            print("Error: style does not exist")
            return None

        if isinstance(normals, bool) and normals:
            # extract manually
            cloud = self.geometry_loader.load_cloud(data_id, as_tensor=False)
            normals = cloud[..., 3:]
            verts = cloud[..., :3]
            if center:
                verts = trimesh_util.normalize_vertices(verts, center=True, scale=False, center_at_centroid=True)

        if normals is not None and not isinstance(normals, bool): # Not pretty but it works
            normals_plot = get_normals_object_for_vis(verts, normals, 0.1)

            pl.add_mesh(
                normals_plot,
                color=self.default_pcn_normals_color,
                opacity=0.6
            )
            normals_plot = add_color_to_pv_object(normals_plot, self.default_pcn_normals_color, lines_only=True)
            multiblock += normals_plot
            # multiblock = add_color_to_pv_object(multiblock, self.default_pcn_normals_color)

        # print(save_path_3d)
        if save_path_3d is not None:
            # print("save to ", save_path_3d)
            # print(multiblock.array_names)
            paths.mkdir(save_path_3d)
            if paths.split_extension(save_path_3d)[1] == ".gltf":
                pl.export_gltf(save_path_3d)
            else:
                multiblock.save(save_path_3d, texture="rgb", binary=False)

        self.set_default_pl_options(pl=pl, text=text, use_case=use_case)

        if save_path is not None:
            image = pl.screenshot(save_path)
            pl.close()
            return image



        return None

    def _add_mesh_to_plot(self, pl, data_id=None, verts=None, faces=None, use_case="default", override_color=None, center=False):
        if data_id is not None:
            verts, _, faces = self.geometry_loader.load_mesh_vef(data_id, as_tensor=False)
            verts = verts[..., :3]
        else:
            assert verts is not None and faces is not None

        if center:
            verts = trimesh_util.normalize_vertices(verts, center=True, scale=True, center_at_centroid=True)

        color = self.default_mesh_color
        if override_color is not None:
            color = override_color

        mesh = convert_to_pv_mesh(verts, faces)
        pl.add_mesh(
            mesh,
            show_edges=use_case == "default", # Here
            color=color,
            ambient=0.2,
            diffuse=0.5,
            specular=0.5,
            specular_power=90,
        )
        # multiblock = pv.MultiBlock()
        mesh = add_color_to_pv_object(mesh, color)
        # multiblock.append(mesh)
        return mesh



    def _add_cloud_to_plot(self, pl, data_id=None, verts=None, use_case="default", normals=False, scan=False, override_color=None, center=False):
        if data_id is not None:
            cloud = self.geometry_loader.load_cloud(data_id, as_tensor=False)
            cloud = cloud[..., :3]
        else:
            cloud = verts

        if center:
            cloud = trimesh_util.normalize_vertices(cloud, center=True, scale=True, center_at_centroid=True)

        color = self.default_cloud_color
        mesh_opacity = 0.3

        if normals:
            color = self.default_pcn_color
        if scan:
            color = self.default_scan_color
            mesh_opacity = 0.1

        if override_color is not None:
            color = override_color

        multiblock = pv.MultiBlock()

        if data_id is not None:
            verts, _, faces = self.geometry_loader.load_stl_vef(data_id, as_tensor=False)
            verts = verts[..., :3]
            mesh = convert_to_pv_mesh(verts, faces)

            # pl.add_mesh(
            #     mesh,
            #     show_edges=False,
            #     color=color,
            #     opacity=mesh_opacity,
            #     smooth_shading=True,
            #     # ambient=0.2,
            #     # diffuse=0.5,
            # )
            # multiblock.append(mesh)

        pl.add_points(
            cloud,
            render_points_as_spheres=True,
            point_size=10,
            color=color
        )

        cloud = pv.PolyData(cloud)
        cloud = add_color_to_pv_object(cloud, color)
        # multiblock.append(cloud)

        return cloud

    def _add_stl_to_plot(self, pl, data_id=None, use_case="default", override_color=None, center=False):
        verts, _, faces = self.geometry_loader.load_stl_vef(data_id, as_tensor=False)
        verts = verts[..., :3]

        if center:
            verts = trimesh_util.normalize_vertices(verts, center=True, scale=True, center_at_centroid=True)

        mesh = convert_to_pv_mesh(verts, faces)

        color = self.default_cad_color
        if override_color is not None:
            color = override_color

        # multiblock = pv.MultiBlock()

        pl.add_mesh(
            mesh,
            show_edges=False,
            color=color,
            split_sharp_edges=True,
            smooth_shading=True,
            ambient=0.2,
            diffuse=0.5,
            specular=1.0,
            specular_power=10,
        )

        mesh = add_color_to_pv_object(mesh, color)
        # multiblock.append(mesh)
        return mesh


    def _add_cad_to_plot(self, pl, data_id=None, verts=None, faces=None, cad_shape=None, use_case="default", override_color=None, center=False):
        if data_id is not None:
            verts, _, faces = self.geometry_loader.load_stl_vef(data_id, as_tensor=False)
            verts = verts[..., :3]
        else:
            assert verts is not None and faces is not None

        if center:
            verts = trimesh_util.normalize_vertices(verts, center=True, scale=True, center_at_centroid=True)

        mesh = convert_to_pv_mesh(verts, faces)

        color = self.default_cad_color
        if override_color is not None:
            color = override_color

        # multiblock = pv.MultiBlock()

        edges = mesh.extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=False,
            feature_angle=30,
            manifold_edges=False,
        )

        pl.add_mesh(
            mesh,
            show_edges=False,
            color=color,
            split_sharp_edges=True,
            smooth_shading=True,
            ambient=0.2,
            diffuse=0.5,
            specular=1.0,
            specular_power=10,
        )
        mesh = add_color_to_pv_object(mesh, color)

        if edges.n_points > 0:
            pl.add_mesh(edges, color='k', line_width=3,
                        render_lines_as_tubes=True,
                        # lighting=False,
                        # ambient=False,
                        # smooth_shading=None,
                        )
            edges = add_color_to_pv_object(edges, "#000000")
            mesh += edges

        return mesh


def load_picture(path, no_background=False):
    picture = cv2.imread(path)

    if no_background:
        THRESHOLD = 240
        intensity = np.mean(picture[..., :3])
        picture[intensity > THRESHOLD, 4] = 0

    return picture

def remove_white(image):
    #image in numpy format
    THRESHOLD = 240
    intensity = np.mean(image[..., :3], axis=2)

    # If there is no fourth dimension, add one
    if image.shape[2] < 4:
        alpha = np.ones(image.shape[:2]) * 255
        image = np.concatenate([image, alpha[..., np.newaxis]], axis=2)
    image[intensity > THRESHOLD, 3] = 0
    return image

def label_to_color(num_classes, classes):
    cmap = colormaps.get_cmap('rainbow')
    classes_normalized = classes * 1.0 / num_classes
    colors = cmap(classes_normalized)
    colors = colors * 255
    return colors

def add_color_to_pv_object(pv_object, color, lines_only=False):
    if isinstance(color, str):
        import matplotlib.colors as mcolors
        # if lines_only:
        #     rgb = mcolors.to_rgba(color, alpha=0)
        # else:
        rgb = mcolors.to_rgb(color)

        rgb_255 = [int(c * 255) for c in rgb]
    else:
        raise NotImplementedError
        # rgb_255 = [int(c * 255) for c in color]

    # Add colors to points
    point_colors = np.tile(rgb_255, (pv_object.n_points, 1)).astype(np.uint8)
    pv_object.point_data['colors'] = point_colors
    pv_object.point_data['rgb'] = point_colors

    # Add colors to cells (lines)
    cell_colors = np.tile(rgb_255, (pv_object.n_cells, 1)).astype(np.uint8)
    pv_object.cell_data['colors'] = cell_colors
    pv_object.cell_data['rgb'] = cell_colors

    return pv_object