import random

class GeneticAlgorithm:
    def __init__(self, candidate, crossover_rate, mutation_rate):
        self.candidate = candidate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def crossover(self, candidate):
        if self.crossover_rate > random.random():
            point = random.randint(1, len(candidate)-1)
            child1 = candidate[point:]
            child2 = candidate[:point]
            new_candidate = child2 + child1
            return new_candidate
            
        else:
            return candidate

    def mutation(self, candidate):
        if self.mutation_rate > random.random():
            return sorted(candidate, key=lambda k: random.random())
        else:
            return candidate

    def generate(self):
        new_candidate = self.mutation(self.crossover(self.candidate))
        return new_candidate