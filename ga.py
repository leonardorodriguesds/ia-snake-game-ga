import numpy as np
import math

class GeneticAlgorithm:
    def __init__(self, individual_class, population_size = 100, num_generations = 100, num_parents = -1, reproduction_probability = 0.5, print_summary = True, random_function = None, dropout = 0.2, mutate_ration=0.1):
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []
        self.individual_class = individual_class
        self.print_summary = print_summary
        self.current_gen = 0
        self.num_parents = num_parents
        self.random_function = random_function
        self.population_kwargs = {}
        self.dropout = dropout
        self.mutate_ration = mutate_ration
        if (self.random_function == None):
            self.random_function = lambda *args,**kwargs: np.random.uniform(*args, **kwargs)
        self.reproduction_probability = reproduction_probability
        if self.num_parents == -1:
            self.num_parents = self.population_size / 2

    def generate_population(self, **kwargs):
        self.population = []
        self.population_kwargs = kwargs
        for i in range(self.population_size):
            self.population.append(self.individual_class(**kwargs))
        return self.population

    def cal_pop_fitness(self):
        fitness = list(map(
            lambda indv: (indv[1].fitness, indv[0]),
            enumerate(self.population)
        ))
        return sorted(fitness, key=lambda i: i[0], reverse=True)

    def is_last_generation(self):
        return self.current_gen == self.num_generations

    def individual_mutate(self, indv):
        return (indv.mutate() if self.random_function(0,1) <= self.mutate_ration else indv)

    def next_generation(self):
        self.current_gen += 1
        fitness = self.cal_pop_fitness()
        new_population = []
        childs = []

        if self.print_summary:
            print(f'top 10 gen {self.current_gen}:\n')
            for j,(_,i) in enumerate(fitness[:10]):
                print(f'{j + 1}º: {self.population[i].fitness}')

        for (_,i) in fitness:
            if (self.random_function(0,1) <= self.reproduction_probability):
                remains = [*self.population]
                del remains[i]
                idx = math.floor(self.random_function(0,len(remains)))
                childs.append(self.population[i].crossover(remains[idx]))
    
        new_population = [*[self.individual_mutate(i) for i in childs], *[self.individual_mutate(self.population[i].copy()) for (_, i) in fitness]]

        new_individuals = []
        dropout_size = math.floor(self.population_size * self.dropout)
        for _ in range(dropout_size):
            new_individuals.append(self.individual_class(**self.population_kwargs))

        new_population[self.population_size-dropout_size:self.population_size] = new_individuals

        self.population = new_population[:self.population_size]
        return self.population