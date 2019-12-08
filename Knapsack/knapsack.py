"""Python3"""

import random
from datasets import items

random.seed(range(100)) #Seed for random usage

#CONFIG

WEIGHT_LIMIT = 60
NO_INDIVIDUAL = 50
MAX_GENE = 30000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3

FINAL_NODE = []
FINAL_WEIGHT = 0


def fitness(target): #FITNESS
    global FINAL_NODE
    global FINAL_WEIGHT
    total_weight = 0
    total_value = 0
    index = 0
    WV = []
    for i in target:
        if index >= len(items):
            break
        if (i == 1):
            total_weight += items[index][0]
            total_value += items[index][1]
            WV.append((items[index][0], items[index][1]))

        index += 1

    if total_weight > WEIGHT_LIMIT:
        return 0
    else:
        if len(FINAL_NODE) < len(WV):
            FINAL_NODE = WV

        if FINAL_WEIGHT < total_weight:
            FINAL_WEIGHT = total_weight
        return total_value


def initialPopulation(amount):
    return [individual() for x in range(0, amount)]


def individual():
    return [random.randint(0, 1) for x in range(0, len(items))]


def mutate(target): #MUTATION
    r = random.randint(0, len(target)-1)
    if target[r] == 1:
        target[r] = 0
    else:
        target[r] = 1


def new_population(pop): #CROSSOVER

    parent_length = int(CROSSOVER_RATE*len(pop)) #SELECTION USING CROSSOVER RATE
    parents = pop[:parent_length] #TOOK SOME OF THE POPULATION TO BECOME PARENT
    nonparents = pop[parent_length:] #AND THE REST ASIDE

    #CROSSOVER FOR SOME NON_PARENTS TO PARENTS
    for np in nonparents:
        if CROSSOVER_RATE/10 > random.random():
            parents.append(np)

    #MUTATE THE PARENTS NODE
    for p in parents:
        if MUTATION_RATE/10 > random.random():
            mutate(p)

    #NEW POPULATION/BREEDINGS
    children = []
    desired_length = len(pop) - len(parents)
    while len(children) < desired_length:
        first_half = pop[random.randint(0, len(parents)-1)]
        second_half = pop[random.randint(0, len(parents)-1)]
        half = int(len(first_half)/2)
        child = first_half[:half] + second_half[half:]
        if MUTATION_RATE > random.random():
            mutate(child)
        children.append(child)

    parents.extend(children)
    return parents


def main():
    generation = 1
    population = initialPopulation(NO_INDIVIDUAL)
    for g in range(0, MAX_GENE):
        fitnessList = []
        print("Generation %d with %d individual" %
              (generation, len(population)))
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        for i in population:
            fitnessList.append(fitness(i))
            print(i)
        population = new_population(population)
        generation += 1

    if generation >= MAX_GENE:
        print(population)
        print("Best Node: %s" % (FINAL_NODE))
        print("Best Value: %s and Best Weight %s" %
              (fitnessList, FINAL_WEIGHT))


if __name__ == "__main__":
    main()
