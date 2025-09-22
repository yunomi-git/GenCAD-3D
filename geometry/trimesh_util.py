import trimesh
import numpy as np
from utils.util import Stopwatch
from tqdm import tqdm
import utils.util as util
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings
import io
from PIL import Image
from PIL import ImageDraw, ImageFont
import robust_laplacian

NO_GAP_VALUE = -1

def mesh_is_valid(mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        return False

    if mesh.bounds is None:
        return False

    if np.isnan(mesh.vertices).any():
        return False

    return True

class MeshAuxilliaryInfo:
    def __init__(self, mesh):
        self.is_valid = mesh_is_valid(mesh)
        if not self.is_valid:
            return

        # try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        # except RuntimeError:
        #     self.is_valid = False
        #     return

        self.mesh = mesh

        self.bound_lower = mesh.bounds[0, :].copy()
        self.bound_upper = mesh.bounds[1, :].copy()
        self.bound_length = self.bound_upper - self.bound_lower

        self.face_centroids = mesh.triangles_center
        self.face_normals = mesh.face_normals
        self.face_areas = mesh.area_faces
        self.surface_area = mesh.area
        self.num_faces = len(self.face_centroids)
        self.faces = mesh.faces
        self.facet_defects = None

        try:
            self.bounding_sphere_center = mesh.bounding_sphere.center.copy()
            self.bounding_sphere_radius = mesh.bounding_sphere.scale
        except Exception as e:
            print("mesh aux error in bounding sphere", e)
            return
        self.volume = mesh.volume

        self.edges = mesh.edges
        self.num_edges = len(self.edges)

        self.vertices = mesh.vertices
        self.num_vertices = len(self.vertices)
        self.vertex_normals = mesh.vertex_normals



    def get_vertices_and_normals(self):
        return self.vertices, self.vertex_normals

    def get_centroids_and_normals(self, return_face_ids=False):
        return self.face_centroids, self.face_normals

    def sample_and_get_normals(self, count=5000, use_weight="even", return_face_ids=False):
        # if use_weight == "even":
        sample_points, face_index = trimesh.sample.sample_surface_even(mesh=self.mesh, count=count)
        if len(sample_points) < count:
            remaining_count = count - len(sample_points)
            extra_points, extra_face_index = trimesh.sample.sample_surface(mesh=self.mesh, count=remaining_count)
            sample_points = np.concatenate((sample_points, extra_points))
            face_index = np.concatenate((face_index, extra_face_index))


        normals = self.face_normals[face_index]

        if return_face_ids:
            return sample_points, normals, face_index
        else:
            return sample_points, normals

    def simple_scan_with_multiplier(self, min_angle, max_angle, num_points, view_angle_pitch, view_angle_yaw, sample_multiplier):
        # generate N random points in: random yaw, random pitch in [min ,max]
        # yaw_list = (np.random.rand(num_points * sample_multiplier) * 2 - 1.0) * np.pi
        points_to_scan = int(num_points * sample_multiplier)
        yaw_list = np.linspace(start=0, stop=np.pi * 2, num=num_points * sample_multiplier)
        pitch_list = np.random.rand(points_to_scan) * (max_angle - min_angle) + min_angle
        view_angle_list_pitch = (np.random.rand(points_to_scan) * 2 - 1.0) * view_angle_pitch / 2.0
        view_angle_list_yaw = (np.random.rand(points_to_scan) * 2 - 1.0) * view_angle_yaw / 2.0

        # These exist on the enclosing sphqere
        r = self.bounding_sphere_radius * 2.5
        object_center = self.bounding_sphere_center.copy()
        camera_center = object_center.copy()
        camera_center[2] = self.bound_lower[2]

        points_x = r * np.cos(yaw_list) * np.cos(pitch_list)
        points_y = r * np.sin(yaw_list) * np.cos(pitch_list)
        points_z = r * np.sin(pitch_list)
        points = np.hstack([points_x[:, np.newaxis], points_y[:, np.newaxis], points_z[:, np.newaxis]])
        points += camera_center

        # point all points to the centroid (origin - centroid).normalize()
        directions_x = r * np.cos(yaw_list + view_angle_list_yaw) * np.cos(pitch_list + view_angle_list_pitch)
        directions_y = r * np.sin(yaw_list + view_angle_list_yaw) * np.cos(pitch_list + view_angle_list_pitch)
        directions_z = r * np.sin(pitch_list + view_angle_list_pitch)
        directions = -np.hstack([directions_x[:, np.newaxis], directions_y[:, np.newaxis], directions_z[:, np.newaxis]])

        # do intersects_first
        face_index, _, intersections = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh).intersects_id(
            ray_origins=points, ray_directions=directions, multiple_hits=False, return_locations=True)

        normals = self.face_normals[face_index]
        return intersections, normals

    def sample_virtual_scan(self, min_angle, max_angle, num_points, view_angle_pitch, view_angle_yaw, return_invalid_scan=False, num_retries=5):
        minimum_points = num_points * 2

        sample_multiplier = 2

        intersections = []
        normals = []
        num_intersections = 0
        for _ in range(num_retries):
            intersections_sample, normals_sample = self.simple_scan_with_multiplier(min_angle, max_angle, num_points, view_angle_pitch, view_angle_yaw, sample_multiplier)

            num_intersections += len(intersections_sample)
            intersections.append(intersections_sample)
            normals.append(normals_sample)

            if num_intersections >= minimum_points:
                break

        intersections = np.concatenate(intersections, axis=0)
        normals = np.concatenate(normals, axis=0)
        if num_intersections < minimum_points:
            if return_invalid_scan:
                return intersections, normals
            else:
                print("not enough points")
                return None, None
        # now try to get matrix
        try:
            # Sample evenly from surface
            _, M = robust_laplacian.point_cloud_laplacian(intersections)
            mass = M.diagonal()
            p = mass
            p /= np.sum(p)

            indices = np.arange(len(intersections))
            selection = np.random.choice(indices, size=num_points, p=p, replace=False)
            intersections_return = intersections[selection]
            normals_return = normals[selection]
        except:
            print("laplacian_error")
            return None, None

        return intersections_return, normals_return


    def get_transformed_mesh(self, scale=1.0, orientation=np.array([0, 0, 0])):
        return MeshAuxilliaryInfo(get_transformed_mesh_trs(self.mesh, scale=scale, orientation=orientation))

def create_transform_matrix(scale=np.array([1, 1, 1]),
                            translation=np.array([0, 0, 0]),
                            orientation=np.array([0, 0, 0])):
    # Order applied: translate, rotate, scale
    # orientation as [x, y, z]
    r = R.from_euler('zyx', [orientation[2], orientation[1], orientation[0]]).as_matrix()
    rot_matrix = np.zeros((4, 4))
    rot_matrix[:3, :3] = r
    rot_matrix[3, 3] = 1.0
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = translation

    transform_matrix = scale_matrix @ (rot_matrix @ trans_matrix)
    return transform_matrix


def get_transformed_mesh_trs(mesh: trimesh.Trimesh, scale=np.array([1, 1, 1]), translation=np.array([0, 0, 0]),
                             orientation=np.array([0, 0, 0])) -> trimesh.Trimesh:
    # Order applied: translate, rotate, scale
    # orientation as [x, y, z]
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale, scale])

    transform_matrix = create_transform_matrix(scale, translation, orientation)
    mesh_copy = mesh.copy()
    mesh_copy = mesh_copy.apply_transform(transform_matrix.astype(np.float32))
    return mesh_copy

def get_largest_submesh(mesh: trimesh.Trimesh):
    if mesh.body_count > 1:
        splits = list(mesh.split(only_watertight=False))
        largest_volume = 0
        largest_submesh = None
        for submesh in splits:
            temp_volume = submesh.volume
            if temp_volume > largest_volume:
                largest_volume = temp_volume
                largest_submesh = submesh
        mesh = largest_submesh
    return mesh

def get_valid_submeshes(mesh: trimesh.Trimesh, sorted=True):
    # Ordered by volume
    valid_meshes = []
    volumes = []
    splits = list(mesh.split(only_watertight=True))
    for submesh in splits:
        mesh_aux = MeshAuxilliaryInfo(submesh)
        if mesh_aux.is_valid:
            valid_meshes.append(submesh)
            volumes.append(mesh_aux.volume)

    # Sort
    if sorted:
        volumes = np.array(volumes)
        sorted_ind = np.argsort(volumes).astype(np.int32)
        valid_meshes = [valid_meshes[i] for i in sorted_ind]
    return valid_meshes

def normalize_mesh(mesh: trimesh.Trimesh, center, normalize_scale) -> trimesh.Trimesh:
    mesh_aux = MeshAuxilliaryInfo(mesh)
    normalization_scale = 1.0
    normalization_translation = np.array([0, 0, 0])
    if center:
        centroid = np.mean(mesh_aux.vertices, axis=0)
        min_bounds = mesh_aux.bound_lower
        normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
    mesh = get_transformed_mesh_trs(mesh, translation=normalization_translation)
    if normalize_scale:
        scale = max(mesh_aux.bound_length)
        normalization_scale = 1.0 / scale

    mesh = get_transformed_mesh_trs(mesh, scale=normalization_scale)
    return mesh

def normalize_vertices(vertices, center, scale, center_at_centroid=False):
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    bounds_length = bounds_max - bounds_min

    normalization_scale = 1.0
    normalization_translation = np.array([0, 0, 0])
    if center:
        centroid = np.mean(vertices, axis=0)
        min_bounds = bounds_min
        if center_at_centroid:
            normalization_translation = -np.array([centroid[0], centroid[1], centroid[2]])
        else:
            normalization_translation = -np.array([centroid[0], centroid[1], min_bounds[2]])
    if scale:
        scale = max(bounds_length)
        normalization_scale = 1.0 / scale

    new_verts = vertices.copy()
    new_verts += normalization_translation
    new_verts *= normalization_scale
    return new_verts


def mirror_surface(mesh: trimesh.Trimesh, plane, process=False):
    mesh_aux = MeshAuxilliaryInfo(mesh)
    faces = mesh_aux.faces
    vertices = mesh_aux.vertices
    # mirror vertices and faces
    mirrored_vertices = vertices.copy()
    if plane == "x":
        mirrored_vertices[:, 0] *= -1
    elif plane == "y":
        mirrored_vertices[:, 1] *= -1
    else:
        mirrored_vertices[:, 2] *= -1
    mirrored_faces = faces.copy()
    mirrored_faces += len(vertices)
    # new_mesh = trimesh.Trimesh(vertices=mirrored_vertices,
    #                            faces=faces,
    #                            merge_norm=True)
    new_verts = np.concatenate([vertices, mirrored_vertices])
    new_faces = np.concatenate([faces, mirrored_faces])
    new_mesh = trimesh.Trimesh(vertices=new_verts,
                               faces=new_faces,
                               merge_norm=True)
    if process:
        new_mesh_aux = MeshAuxilliaryInfo(new_mesh)
        return new_mesh_aux.mesh
    else:
        return new_mesh

def repair_missing_mesh_values(mesh, vertex_ids, values, max_iterations=2, iteration=0):
    # Samples may sometimes be missing values. If so, interpolate value using nearby values
    mesh_aux = MeshAuxilliaryInfo(mesh)
    num_vertices = len(mesh_aux.vertices)

    # First create a list of values for all vertices. Missing values are nan
    input_values_padded = np.empty(num_vertices)
    input_values_padded[:] = np.nan
    input_values_padded[vertex_ids] = values

    # To return
    repaired_values = np.zeros(num_vertices)

    # Grab values of vertices connected to missing vertices
    vertex_connection_ids = mesh.vertex_neighbors # list
    missing_ids = np.delete(np.arange(num_vertices), vertex_ids).astype(np.int32)
    missing_connection_ids = [vertex_connection_ids[i] for i in missing_ids] # list
    # Construct the missing values
    missing_values = np.empty(len(missing_ids))
    for i in range(len(missing_ids)):
        connections_ids = missing_connection_ids[i]
        connection_values = input_values_padded[connections_ids]
        connection_values = connection_values[~np.isnan(connection_values)]
        missing_values[i] = np.mean(connection_values)

    # Average and set
    repaired_values[vertex_ids] = values
    repaired_values[missing_ids] = missing_values

    if np.isnan(repaired_values).any() and iteration < max_iterations:
        valid_indices = np.arange(num_vertices)[~np.isnan(repaired_values)]
        repaired_values = repair_missing_mesh_values(mesh, vertex_ids=valid_indices,
                                                     values=repaired_values[valid_indices],
                                                     max_iterations=max_iterations, iteration=iteration+1)
    return repaired_values


##### Visualization

def show_sampled_values(mesh, points, values, normalize=True, scale=None, alpha=0.8):
    s = trimesh.Scene()
    set_default_camera(s, mesh, isometric=True)
    if len(points) > 0:
        if normalize:
            values = util.normalize_minmax_01(values)
        elif scale is not None:
            values[values > scale[1]] = scale[1]
            values[values < scale[0]] = scale[0]

        cmapname = 'jet'
        cmap = plt.get_cmap(cmapname)
        colors = 255.0 * cmap(values)
        colors[:, 3] = int(alpha * 255)
        point_cloud = trimesh.points.PointCloud(vertices=points,
                                                colors=colors)
        s.add_geometry(point_cloud)
    if mesh is not None:
        s.add_geometry(mesh)
    s.show()

def show_mesh_with_normals(mesh, points, normals):
    s = trimesh.Scene()
    set_default_camera(s, mesh, isometric=True)
    if len(points) > 0:
        colors = np.array([0, 0, 255, 255])
        point_cloud = trimesh.points.PointCloud(vertices=points,
                                                colors=colors)
        s.add_geometry(point_cloud)
        for i in range(len(points)):
            line = trimesh.load_path(np.array([points[i], normals[i]/3 + points[i]]))
            s.add_geometry(line)
    s.add_geometry(mesh)
    s.show()

def show_mesh_with_facet_colors(mesh, values: np.ndarray, normalize=True):
    s = trimesh.Scene()
    set_default_camera(s, mesh)
    cmapname = 'jet'
    cmap = plt.get_cmap(cmapname)
    empty_color = np.array([100, 100, 100, 255])

    if normalize:
        values[values != NO_GAP_VALUE] = util.normalize_minmax_01(values[values != NO_GAP_VALUE])
    mesh.visual.face_colors = cmap(values)
    mesh.visual.face_colors[values == NO_GAP_VALUE] = empty_color

    s.add_geometry(mesh)
    s.show()

def set_default_camera(scene: trimesh.Scene, mesh: trimesh.Trimesh, isometric=False):
    width = mesh.bounds[1, 1] - mesh.bounds[0, 1]
    height = mesh.bounds[1, 2] - mesh.bounds[0, 2]
    length = mesh.bounds[1, 0] - mesh.bounds[0, 0]
    centroid = mesh.centroid


    orientation = np.array([np.pi / 2, 0, 0])
    if isometric:
        radius = np.sqrt(width ** 2 + height ** 2 + length ** 2)
        camera_offset = np.array([0, -radius * 1.5, 0])
        isometric_orientation = np.array([-np.pi/6, -np.pi / 3.7, -np.pi/8])
        isometric_transform = R.from_euler('zyx', [-np.pi / 3.7, 0, -np.pi/6]).as_matrix()
        orientation_transform = create_transform_matrix(orientation=orientation + isometric_orientation)
        translation_transform = create_transform_matrix(translation=centroid + isometric_transform @ camera_offset)
    else:
        radius = np.sqrt(height ** 2 + length ** 2)
        camera_offset = np.array([0, -radius * 1.3, 0])
        orientation_transform = create_transform_matrix(orientation=orientation)
        translation_transform = create_transform_matrix(translation=centroid + camera_offset)
    scene.camera_transform = translation_transform @ orientation_transform

def show_mesh(mesh, isometric=True):
    s = trimesh.Scene()
    set_default_camera(s, mesh, isometric=isometric)
    s.add_geometry(mesh)
    s.show()

def show_meshes(meshes):
    s = trimesh.Scene()
    set_default_camera(s, meshes[0])
    for mesh in meshes:
        s.add_geometry(mesh)
    s.show()

def get_mesh_picture(mesh: trimesh.Trimesh, resolution=1080, isometric=True):
    s = trimesh.Scene()
    set_default_camera(s, mesh, isometric=isometric)
    s.add_geometry(mesh)

    data = s.save_image(resolution=(resolution, resolution), visible=True)
    image = Image.open(io.BytesIO(data))
    return image


def save_mesh_picture(mesh: trimesh.Trimesh, name, resolution=1080, isometric=True, text=None):
    image = get_mesh_picture(mesh, resolution, isometric)
    if text is not None:
        ImageDraw.Draw(
            image  # Image
        ).text(
            (0, 0),  # Coordinates
            text,  # Text
            (0, 0, 0),  # Color,
            font=ImageFont.load_default(resolution // 10)
        )
    image.save(f"{name}", "PNG")


def show_mesh_with_orientation(mesh):
    mesh_aux = MeshAuxilliaryInfo(mesh)
    colors = util.direction_to_color(mesh_aux.face_normals)
    mesh.visual.face_colors = colors
    s = trimesh.Scene()
    set_default_camera(s, mesh)
    s.add_geometry(mesh)
    s.show()

def show_mesh_with_z_normal(mesh):
    mesh_aux = MeshAuxilliaryInfo(mesh)
    colors = util.z_normal_mag_to_color(mesh_aux.face_normals)
    mesh.visual.face_colors = colors

    s = trimesh.Scene()
    set_default_camera(s, mesh)
    s.add_geometry(mesh)
    s.show()

