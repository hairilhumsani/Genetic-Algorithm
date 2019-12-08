import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import math
import VRP.dataset


class AnimatedVisual:
    def animateTSP(history, points):
        key_frames_mult = len(history) // 1500
        fig, ax = plt.subplots()
        line, = plt.plot([], [], lw=2)

        def init():
            x = [points[i][0] for i in history[0]]
            y = [points[i][1] for i in history[0]]
            plt.plot(x, y, 'co')

            extra_x = (max(x) - min(x)) * 0.05
            extra_y = (max(y) - min(y)) * 0.05
            ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
            ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

            line.set_data([], [])
            return line

        def update(frame):
            x = [points[i, 0] for i in history[frame] + [history[frame][0]]]
            y = [points[i, 1] for i in history[frame] + [history[frame][0]]]
            line.set_data(x, y)
            return line

        ani = FuncAnimation(fig, update, frames=range(
            0, len(history), key_frames_mult), init_func=init, interval=3, repeat=False)

        plt.show()


class NodeGenerator:
    def __init__(self, width, height, nodesNumber):
        self.width = width
        self.height = height
        self.nodesNumber = nodesNumber

    def generate(self):
        xs = np.random.randint(self.width, size=self.nodesNumber)
        ys = np.random.randint(self.height, size=self.nodesNumber)

        return np.column_stack((VRP.dataset.x_data, VRP.dataset.y_data))


class SimulatedAnnealing:
    def __init__(self, coords, temp, alpha, stopping_temp, stopping_iter):
        self.coords = coords
        self.sample_size = len(coords)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.dist_matrix = TSP_UTILS.vectorToDistMatrix(coords)
        self.curr_solution = TSP_UTILS.nearestNeighbourSolution(
            self.dist_matrix)
        self.best_solution = self.curr_solution

        self.solution_history = [self.curr_solution]

        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]

        print('Intial weight: ', self.curr_weight)
    
    def weight(self, sol):
        return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])

    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        candidate_weight = self.weight(candidate)
        print(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    def anneal(self):
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)

        print('Minimum weight: ', self.min_weight)
        print('Improvement: ',
              round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')

    def animateSolutions(self):
        AnimatedVisual.animateTSP(self.solution_history, self.coords)

    def plotLearning(self):
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight,
                                color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], [
                   'Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Iteration')
        plt.show()


class TSP_UTILS:
    def vectorToDistMatrix(coords):
        '''
        Create the distance matrix
        '''
        return np.sqrt((np.square(coords[:, np.newaxis] - coords).sum(axis=2)))

    def nearestNeighbourSolution(dist_matrix):
        '''
        Computes the initial solution (nearest neighbour strategy)
        '''
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


def main():
    '''set the simulated annealing algorithm params'''
    temp = 1000
    stopping_temp = 0.00000001
    alpha = 0.9995
    stopping_iter = 10000000

    '''set the dimensions of the grid'''
    size_width = 200
    size_height = 200

    '''set the number of nodes'''
    population_size = 70

    '''generate random list of nodes'''
    nodes = NodeGenerator(size_width, size_height, population_size).generate()

    '''run simulated annealing algorithm with 2-opt'''
    sa = SimulatedAnnealing(nodes, temp, alpha, stopping_temp, stopping_iter)
    sa.anneal()

    '''animate'''
    sa.animateSolutions()

    '''show the improvement over time'''
    sa.plotLearning()


if __name__ == "__main__":
    main()