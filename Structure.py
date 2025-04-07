from Voronoi import Voronoi


class Structure:
    def __init__(self, voronoi: Voronoi, value: float):
        self.voronoi = voronoi
        self.structure = self.voronoi.structure
        self.value = value

    def __lt__(self, other):
        if self.structure != other.structure:
            return self.structure < other.structure
        return self.value < other.value
