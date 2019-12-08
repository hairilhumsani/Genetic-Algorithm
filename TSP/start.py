import random
import math

from population import Population
from tsp_ga import GeneticAlgorithm
from utils import UTILS
from animate import AnimatedVisual


class Start:
    def __init__(self, coords, pop_size):
        self.coords = coords
        self.pop_size = pop_size
        self.sample_size = len(coords)

        self.pop_init = Population(self.coords, self.pop_size)
        self.distance_matrix = self.pop_init.distance_matrix()

        self.current_solutions = self.pop_init.nearest_neighbours()
        self.current_solution = self.current_solutions[0]
        self.best_solution = self.current_solution
        self.solution_history = [self.current_solution]

        self.current_distance = self.pop_init.distance(
            self.distance_matrix, self.current_solution)
        self.initial_distance = self.current_distance
        self.min_distance = self.current_distance

    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.current_distance) / self.pop_size)

    def weight(self, sol):
        return UTILS.distance(self.distance_matrix, sol)

    def accept(self, candidate):
        candidate_weight = self.weight(candidate)
        if candidate_weight < self.current_distance:
            self.current_distance = candidate_weight
            self.current_solution = candidate
            if candidate_weight < self.min_distance:
                self.min_distance = candidate_weight
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.current_distance = candidate_weight
                self.current_solution = candidate

        self.new_solution.append(self.best_solution)

    def ga(self, generation, crossover, mutation):
        generation_at = 1
        for x in range(0, generation):
            for i in self.current_solutions:
                self.new_solution = []
                i = GeneticAlgorithm(
                    i, crossover, mutation).generate()
                candidate = list(i)
                l = random.randint(2, self.sample_size - 1)
                i = random.randint(0, self.sample_size - l)

                candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

                self.accept(candidate)
                self.solution_history.append(self.current_solution)
            self.current_solutions = self.new_solution
            print("Generation %d with %d population" %
                  (generation_at, self.pop_size))
            generation_at += 1
        print("Initial distance: ", self.initial_distance)
        print("New distance:", self.min_distance)

    def animate(self):
        AnimatedVisual.animateTSP(self.solution_history, self.coords)
