import numpy as np
import matplotlib.pyplot as plt
from spherical_geometry.polygon import SphericalPolygon
from scipy.spatial import geometric_slerp
from Voronoi import spherical_to_cartesian, haversine, Voronoi
from numpy import ndarray


class Polygon:
    def __init__(self, vertices: ndarray, regions: list):
        self.size = len(regions)
        self.latv = vertices[:, 0]
        self.lonv = vertices[:, 1]
        self.vertices = [spherical_to_cartesian(lon, lat) for [lat, lon] in vertices]
        self.regions = regions
        self.areas = np.zeros(self.size)
        self.perimeter = 0
        self.edges = []
        for i in range(self.size):
            polygon = SphericalPolygon(np.array([self.vertices[j] for j in self.regions[i]]))
            self.areas[i] = polygon.area()
        for region in self.regions:
            n = len(region)
            for i in range(n):
                dis = haversine(self.lonv[region[i]], self.latv[region[i]],
                                self.lonv[region[(i + 1) % n]], self.latv[region[(i + 1) % n]])
                self.perimeter = self.perimeter + dis
                self.edges.append(dis)

    def savefig(self, filename, unit_sphere=1):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        t_vals = np.linspace(0, 1, 2000)

        if unit_sphere:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='lightgray', alpha=0.2, linewidth=0)
        
        # ax.scatter(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2], color='k', s=10)

        for region in self.regions:
            n = len(region)
            for i in range(n):
                start = self.vertices[region[i]]
                end = self.vertices[region[(i + 1) % n]]
                result = geometric_slerp(start, end, t_vals)
                ax.plot(result[:, 0], result[:, 1], result[:, 2], c='k')

        ax.set_box_aspect([1,1,1])  # 保证坐标轴等比例
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        plt.savefig(
            f'figures/{filename}.svg', 
            format='svg', 
            bbox_inches='tight', 
            dpi=600, 
            pad_inches=0.1, 
            transparent=False
        )


def voronoi2polygon(voronoi: Voronoi) -> Polygon:
    return Polygon(np.array([list(pair) for pair in zip(*[voronoi.latv, voronoi.lonv])]), 
                   voronoi.sv.regions)
