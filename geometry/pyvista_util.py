import numpy as np
import pyvista as pv
from geometry import trimesh_util


def convert_to_pv_mesh(vertices, faces):
    pad = 3.0 * np.ones((len(faces), 1))
    faces = np.concatenate((pad, faces), axis=1)
    faces = np.hstack(faces).astype(np.int64)
    mesh = pv.PolyData(vertices, faces)
    return mesh

def get_normals_object_for_vis(vertices, normals, normals_scale):
    vertices = vertices[..., :3]
    normals_vertices = vertices + normals * normals_scale
    num_verts = len(vertices)
    points = np.concatenate([vertices, normals_vertices])
    lines = np.hstack([[2, i, i+num_verts] for i in range(num_verts)])
    mesh = pv.PolyData(points, lines=lines)
    return mesh

def show_mesh(vertices, faces, save_name=None):
    mesh = convert_to_pv_mesh(vertices, faces)

    default_size = 600

    pl = pv.Plotter(window_size=[default_size, default_size])
    actor1 = pl.add_mesh(
        mesh,
        show_edges=True,
    )

    if save_name is None:
        pl.show()
    else:
        pl.show()
        return pl.screenshot(save_name)


def plot_mesh_default(pl, vertices, faces, name=None):
    mesh = convert_to_pv_mesh(vertices, faces)
    actor1 = pl.add_mesh(
        mesh,
        show_edges=True,
    )
    if name is not None:
        pl.add_text(name, color='black')
    return actor1


def plot_cloud_default(pl, cloud, name=None):
    actor1 = pl.add_points(
        cloud,
        render_points_as_spheres=True,
        point_size=3,
    )
    if name is not None:
        pl.add_text(name, color='black')
    return actor1

def show_geometries_in_grid(geometries, names=None, r=4, c=4, save_name="temp", transpose=False):
    num_data = len(geometries)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data/ num_visualized))
    default_size = 800

    if transpose:
        temp = r
        r = c
        c = temp

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c), window_size=[default_size * c, default_size * r])
        for ri in range(r):
            for ci in range(c):
                if not transpose:
                    idx = i * r * c + ri * c + ci
                else:
                    idx = i * r * c + r * ci + ri
                if idx >= num_data:
                    break
                pl.subplot(ri, ci)

                geometry = geometries[idx]
                if geometry is None:
                    continue
                name = None
                if names is not None:
                    name = names[idx]
                if len(geometry) == 2:
                    # mesh
                    # actor = plot_mesh_default(pl, geometry[0], geometry[1], name)
                    mesh = convert_to_pv_mesh(geometry[0], geometry[1])
                    actor1 = pl.add_mesh(
                        mesh,
                        show_edges=True,
                    )
                    if name is not None:
                        pl.add_text(name, color='black')
                else:
                    actor = plot_cloud_default(pl, (geometry[0]), name)
                # pl.show_bounds(grid=True, all_edges=False,  font_size=10)

        pl.link_views()
        pl.show()
        pl.screenshot(save_name + ".png")



def show_meshes_in_grid(meshes, r=4, c=4):
    num_data = len(meshes)
    num_visualized = c * r
    num_iterations = int(np.ceil(num_data / num_visualized))

    for i in range(num_iterations):
        pl = pv.Plotter(shape=(r, c))
        for ri in range(r):
            for ci in range(c):
                idx = i * r * c + ri * c + ci
                if idx >= num_data:
                    break
                mesh_vertices = meshes[idx].vertices
                mesh_faces = meshes[idx].faces
                # mesh_labels = labels[idx]
                mesh = convert_to_pv_mesh(mesh_vertices, mesh_faces)
                pl.subplot(ri, ci)
                actor = pl.add_mesh(
                    mesh,
                )
                pl.show_bounds(grid=True, all_edges=False,  font_size=10)

        pl.link_views()
        pl.show()

def show_mesh_z_mag(trimesh_mesh):
    mesh_aux = trimesh_util.MeshAuxilliaryInfo(trimesh_mesh)

    def get_angle_magnitudes(direction):
        horiz_dir = np.sqrt(direction[:, 1]**2 + direction[:, 0] ** 2)
        vert_dir = direction[:, 2]
        pitch = np.arctan2(vert_dir, horiz_dir)  # range -pi/2, pi/2
        pitch = np.abs(pitch) / (np.pi / 2) # range is 0, pi/2. change to 0, 1
        return pitch

    angle_magnitudes = get_angle_magnitudes(mesh_aux.face_normals)

    mesh = convert_to_pv_mesh(mesh_aux.vertices, mesh_aux.faces)

    default_size = 600
    pl = pv.Plotter(window_size=[default_size, default_size])
    actor1 = pl.add_mesh(
        mesh,
        show_edges=True,
        scalars=angle_magnitudes.flatten(),
        show_scalar_bar=False,
        # scalar_bar_args={'title': 'Actual',
        #                  'n_labels': 3},
        # clim=[min_value, max_value]
    )
    # pl.add_text('Actual', color='black')
    pl.show_bounds()
    actor1.mapper.lookup_table.cmap = 'jet'

    pl.show()
