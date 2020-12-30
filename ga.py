import numpy as np

def cal_pop_fitness(snakes):
    fitness_list = np.array(list(map(
        lambda s: s.fitness,
        snakes
    )))
    liveOn_list = np.array(list(map(
        lambda s: s.liveOn,
        snakes
    )))
    fitness = list(map(
        lambda s: (s.fitness - np.min(fitness_list) / (np.max(fitness_list) - np.min(fitness_list)) if np.max(fitness_list) - np.min(fitness_list) > 0 else 0 ) * 0.95 + ((((s.liveOn - np.min(liveOn_list)) / (np.max(liveOn_list) - np.min(liveOn_list))) if np.max(liveOn_list) - np.min(liveOn_list) > 0 else 0) * 0.05),
        snakes
    ))
    return np.array(fitness)


def select_mating_pool(pop, fitness, num_parents):

    parents = []

    for _ in range(num_parents):

        max_fitness_idx = np.where(fitness == np.max(fitness))

        max_fitness_idx = max_fitness_idx[0][0]

        parents.append(pop[max_fitness_idx])

        fitness[max_fitness_idx] = -99999999999

    return parents