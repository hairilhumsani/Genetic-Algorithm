import numpy as np

class NodeGenerator:
    def __init__(self, x_array, y_array):
        self.x_array = x_array
        self.y_array = y_array

    def coords(self):
        coords = np.column_stack((self.x_array, self.y_array))
        return coords