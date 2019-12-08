import random
import math
import dataset

from nodeGenerator import NodeGenerator
from start import Start

def main():
    mutation = 0.3
    crossover = 0.8
    pop_size = 10
    generation_size = 30000

    coords = NodeGenerator(dataset.x_data, dataset.y_data).coords()
    start = Start(coords,pop_size)
    start.ga(generation_size,crossover,mutation)
    start.animate()


if __name__ == "__main__":
    main()
