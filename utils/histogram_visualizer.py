import matplotlib.pyplot as plt
import numpy as np
import math

class MultiHistogram:
    def __init__(self, data=None, bins=None, density=True, num_figs=1, colors=None):
        # Either choose data/bins or histogram
        self.histograms = []
        self.x = []
        for data_slice in data:
            hist, bin_edges = np.histogram(data_slice, bins, density=density)
            x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            self.histograms.append(hist)
            self.x.append(x)

        self.x = np.array(self.x)
        self.histograms = np.array(self.histograms)

        self.std = np.std(self.histograms)
        self.mean = np.mean(self.histograms)
        self.median = np.median(self.histograms)
        self.max = np.max(self.histograms)
        clipped_histograms = np.copy(self.histograms)
        clipped_histograms = clipped_histograms[clipped_histograms < self.mean + 4 * self.std]
        self.clipped_mean = np.mean(clipped_histograms)
        self.clipped_std = np.std(clipped_histograms)

        self.bins = bins
        self.num_plots = len(self.histograms)
        self.num_figs = num_figs
        self.side_length = int(np.sqrt(self.num_figs))
        self.colors = colors

    def plot(self, fig, idx=1):
        ax = fig.add_subplot(self.side_length, int(np.ceil(self.num_figs / self.side_length)), idx, projection='3d')

        for i in range(self.num_plots):
            x = self.x[i]
            hist = self.histograms[i]
            width = (max(x) - min(x)) / self.bins
            ax.bar(x, hist, zs=i, width=width, zdir='y',  alpha=0.3, color=self.colors[i])
            ax.set_zlim(zmax=self.clipped_std * 4 + self.clipped_mean)
            ax.set_zlabel("density")
            ax.set_xlabel("parameter value")
            ax.set_ylabel("sequence length")
            # ax.view_init(elev=0, azim=-90)

        return ax

def _hist_test():
    num_plots = 8
    data = []
    for _ in range(num_plots):
        data_i = np.random.normal(loc=np.random.rand(), scale=np.random.rand(), size=500)
        data.append(data_i)

    multi = MultiHistogram(data=data, bins=10)



class VectorPlotter:
    def __init__(self, data, labels=None):
        # data is (N, D) where D <= 3
        self.data = data
        self.num_dim = data.shape[1]

        color_map = plt.get_cmap('gist_rainbow')
        self.basis_colors = color_map(np.linspace(start=0, stop=1, num=self.num_dim+1))
        self.basis_positions = self.create_equilateral_polygon(num_sides=self.num_dim, side_length=1)
        self.labels = labels

    def create_equilateral_polygon(self, num_sides, side_length, center_x=0, center_y=0):
        if num_sides < 3:
            raise ValueError("Polygon must have at least 3 sides")

        vertices = []
        for i in range(num_sides):
            angle = 2 * np.pi * i / num_sides
            x = center_x + side_length / (2 * np.tan(np.pi / num_sides)) * np.cos(angle)
            y = center_y + side_length / (2 * np.tan(np.pi / num_sides)) * np.sin(angle)
            vertices.append(np.array([x, y]))
        return np.array(vertices)

    def interpolate_point(self, vector):
        color = 0
        position = 0
        for i in range(self.num_dim):
            color += vector[i] * self.basis_colors[i]
            position += vector[i] * self.basis_positions[i]
        return color, position

    def plot(self, ax):
        colors = []
        positions = []
        for vector in self.data:
            color, position = self.interpolate_point(vector)
            colors.append(color)
            positions.append(position)

        colors = np.clip(np.array(colors), 0, 1)
        positions = np.array(positions)
        ax.scatter(x=positions[:, 0], y=positions[:, 1], c=colors)

        # Plot grid
        grid_vertices = np.concatenate([self.basis_positions, self.basis_positions[0, np.newaxis]])
        ax.plot(grid_vertices[:, 0], grid_vertices[:, 1], c='k')

        # Plot bases
        for i in range(self.num_dim):
            color = self.basis_colors[i]
            position = self.basis_positions[i]
            label = None
            if self.labels is not None:
                label = self.labels[i]
            ax.scatter(x=position[0], y=position[1], color=color, s=100, label=label)

        ax.axis('equal')
        ax.legend()

def _vector_test():
    num_dim = 3
    data = np.random.rand(100, num_dim)
    data = data / np.repeat(np.sum(data, axis=1)[..., np.newaxis], axis=1, repeats=num_dim)
    fig, ax = plt.subplots()

    plotter = VectorPlotter(data, ["arc", "line", "circle", "etc"])
    plotter.plot(ax)
    plt.show()
