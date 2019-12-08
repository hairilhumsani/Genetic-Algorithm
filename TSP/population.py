from utils import UTILS

class Population:
    def __init__(self, coords, amount):
        self.coords = coords
        self.amount = amount

    def distance_matrix(self):
        return UTILS.distance_matrix(self.coords)

    def nearest_neighbours(self):
        return [UTILS.nearest_neighbour(self.distance_matrix())for i in range(0, self.amount)]

    def distance(self, distance_matrix, candidate):
        return UTILS.distance(distance_matrix, candidate)