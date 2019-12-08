import numpy as np
import random

class UTILS:
    def distance_matrix(coords):
        return np.sqrt((np.square(coords[:, np.newaxis]-coords).sum(axis=2)))

    def nearest_neighbour(dist_matrix):
        node = random.randrange(len(dist_matrix))
        result = [node]

        nodes_to_visit = list(range(len(dist_matrix)))
        nodes_to_visit.remove(node)

        while nodes_to_visit:
            nearest_node = min([(dist_matrix[node][j], j)
                                for j in nodes_to_visit], key=lambda x: x[0])
            node = nearest_node[1]
            nodes_to_visit.remove(node)
            result.append(node)

        return result

    def distance(distance_matrix, result):
        return sum([distance_matrix[i, j]
                    for i, j in zip(result, result[1:] + [result[0]])])