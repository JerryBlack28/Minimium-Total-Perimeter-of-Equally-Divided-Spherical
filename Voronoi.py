import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, geometric_slerp


def spherical_to_cartesian(lon, lat, r=1):  # lon: 经度, lat: 维度
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    xy2 = x**2 + y**2
    r = np.sqrt(xy2 + z**2)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(xy2))
    return lon, lat, r


def haversine(lon1, lat1, lon2, lat2, radius=1, rad=1):
    # distance between latitudes and longitudes
    dLat = min(math.fabs(lat2 - lat1),
               (2 * np.pi if rad else 360) - math.fabs(lat2 - lat1))
    dLon = math.fabs(lon2 - lon1)
    if not rad:
        dLat = dLat * np.pi / 180
        dLon = dLon * np.pi / 180

    # convert to radians
    if not rad:
        lat1 = lat1 * np.pi / 180.0
        lat2 = lat2 * np.pi / 180.0
 
    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) +
         pow(np.sin(dLon / 2), 2) *
         np.cos(lat1) * np.cos(lat2))
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


class Voronoi:
    def __init__(self, points, radius=1, center=np.array([0, 0, 0])):
        self.size = np.size(points, 0)
        self.radius = radius
        self.center = center
        self.points = points
        self.lonp = np.array([])
        self.latp = np.array([])
        self.lonv = np.array([])
        self.latv = np.array([])
        self.structure = []
        num = 0
        for x, y, z in self.points:
            lon, lat, _ = cartesian_to_spherical(x, y, z)
            self.lonp = np.insert(self.lonp, num, lon)
            self.latp = np.insert(self.latp, num, lat)
            num += 1
        self.sv = SphericalVoronoi(self.points, self.radius, self.center)
        num = 0
        for x, y, z in self.sv.vertices:
            lon, lat, _ = cartesian_to_spherical(x, y, z)
            self.lonv = np.insert(self.lonv, num, lon)
            self.latv = np.insert(self.latv, num, lat)
            num += 1
        self.areas = self.sv.calculate_areas()
        self.max_area = max(self.areas)
        self.perimeter = 0
        self.edges = []
        for region in self.sv.regions:
            n = len(region)
            self.structure.append(n)
            for i in range(n):
                dis = haversine(self.lonv[region[i]], self.latv[region[i]],
                                self.lonv[region[(i + 1) % n]], self.latv[region[(i + 1) % n]])
                self.perimeter = self.perimeter + dis
                self.edges.append(dis)
        self.structure.sort()
                

    def draw(self, unit_sphere=1, generator_points=1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        t_vals = np.linspace(0, 1, 2000)

        if unit_sphere:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='y', alpha=0.1)
        
        if generator_points:
            self.sv.sort_vertices_of_regions()
            ax.scatter(self.points[:, 0], self.points[:, 1], 
                       self.points[:, 2], c='b')
            
        for region in self.sv.regions:
            n = len(region)
            for i in range(n):
                start = self.sv.vertices[region][i]
                end = self.sv.vertices[region][(i + 1) % n]
                result = geometric_slerp(start, end, t_vals)
                ax.plot(result[:, 0], result[:, 1], result[:, 2], c='k')

        ax.azim = 10
        ax.elev = 40
        _ = ax.set_xticks([])
        _ = ax.set_yticks([])
        _ = ax.set_zticks([])
        fig.set_size_inches(4, 4)
        plt.gca().set_aspect('equal')
        plt.show()

