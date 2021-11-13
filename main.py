# conda create --name deap-env python deap pandas tqdm matplotlib -c conda-forge

from datetime import datetime

import numpy
from deap import base, creator, tools
import random

# board configuration
rows_size = [3, 4, 2, 2, 4, 3]
perm_hor_sum = [16, 20, 6, 16, 19, 22]
perm_ver_sum = [4, 33, 3, 16, 35, 8]
cols_idxs = [[(2, 0), (4, 0)], [(0, 0), (1, 0), (2, 1), (4, 1), (5, 0)], [(0, 1), (1, 1)], [(4, 2), (5, 1)],
             [(0, 2), (1, 2), (3, 0), (4, 3), (5, 2)], [(1, 3), (3, 1)]]

# rows_size = [2, 3, 2,
#              8, 2,
#              3, 2, 3,
#              2, 2, 4,
#              2, 2, 2, 2,
#              2, 4, 2,
#              2, 2, 2,
#              2, 2, 5,
#              5, 2, 2,
#              2, 2, 2,
#              2, 4, 2,
#              2, 2, 2, 2,
#              4, 2, 2,
#              3, 2, 3,
#              2, 8,
#              2, 3, 2]
# perm_hor_sum = [16, 14, 6,
#                 39, 12,
#                 10, 13, 24,
#                 13, 13, 11,
#                 17, 14, 3, 9,
#                 12, 17, 12,
#                 3, 14, 6,
#                 7, 3, 34,
#                 34, 5, 17,
#                 15, 8, 4,
#                 4, 18, 15,
#                 11, 5, 17, 13,
#                 21, 13, 16,
#                 21, 8, 18,
#                 11, 43,
#                 17, 24, 15]
# perm_ver_sum = []

allele_min_range = 1
allele_max_range = 10

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
stats = tools.Statistics(lambda ind: ind.fitness.values)


def init_individual():
    rows_perm = []
    for row_i, row_size in enumerate(rows_size):
        toolbox.register(f'row{row_i}', random.sample, range(allele_min_range, allele_max_range), row_size)
        rows_perm.append(eval(f'toolbox.row{row_i}'))

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     rows_perm, n=len(rows_perm))


def init_population():
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def init_evaluator(rows_weight=0.33, cols_weight=0.33, cols_dup_weight=0.34):
    # arithmetic sequence sum
    def dup_penalty(n, d=5):
        return (d * n * (n - 1)) / 2

    def eval_fitness(ind):
        rows_penalty = 0
        cols_penalty = 0
        cols_dup_penalty = 0
        # rows penalty
        for row_i, row_size in enumerate(rows_size):
            ind_row_sum = sum(ind[row_i])
            row_penalty = abs(ind_row_sum - perm_hor_sum[row_i])
            rows_penalty += row_penalty

        # cols penalty
        cols = [[ind[row_i][cell_i] for row_i, cell_i in col_idx] for col_idx in cols_idxs]
        for col_i, col_vals in enumerate(cols):
            ind_col_sum = sum(col_vals)
            col_penalty = abs(ind_col_sum - perm_ver_sum[col_i])
            cols_penalty += col_penalty

            num_occurrences = [0] * 9
            for col_val in col_vals:
                num_occurrences[col_val - 1] += 1
            col_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            cols_dup_penalty += col_dup_penalty

        total_penalty = rows_weight * rows_penalty + cols_weight * cols_penalty + cols_dup_weight * cols_dup_penalty
        return total_penalty

    toolbox.register("evaluate", eval_fitness)


def init_crossovers():
    toolbox.register("mate", tools.cxOnePoint)


def init_mutations(shuffle_bit_prob=0.05, replace_prob=0.5):
    def shuffle_mutation(individual, indpb):
        mutated_ind = random.randrange(0, len(individual))
        new_indi = toolbox.clone(individual)
        new_indi[mutated_ind], = tools.mutShuffleIndexes(new_indi[mutated_ind], indpb)
        return new_indi

    def replace_mutation(individual):
        mutated_row_ind = random.randrange(0, len(individual))
        mutated_row = individual[mutated_row_ind]
        mutated_ind = random.randrange(0, len(mutated_row))
        num_to_choice = list(set(range(1, 10)) - set(mutated_row))
        num_to_insert = random.choice(num_to_choice)

        new_indi = toolbox.clone(individual)
        new_indi[mutated_row_ind][mutated_ind] = num_to_insert
        return new_indi

    def mutation(individual):
        if random.random() < replace_prob:
            return replace_mutation(individual)

        return shuffle_mutation(individual, shuffle_bit_prob)

    toolbox.register("mutate", mutation)


def init_selections(tournsize=3):
    toolbox.register("select", tools.selTournament, tournsize=tournsize)


def init_statistics():
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("median", numpy.median)


def init_GA():
    init_individual()
    init_population()
    init_selections()
    init_evaluator()
    init_crossovers()
    init_mutations()
    init_statistics()


def create_offsprings(parents, cross_pb, mutation_pb):
    offsprings = [toolbox.clone(indi) for indi in parents]
    random.shuffle(offsprings)

    # Cross-Over
    for i in range(0, len(offsprings), 2):
        if random.random() < cross_pb:
            offsprings[i], offsprings[i + 1] = toolbox.mate(offsprings[i], offsprings[i + 1])
            del offsprings[i].fitness.values, offsprings[i + 1].fitness.values

    # Mutation
    for i in range(len(offsprings)):
        if random.random() < mutation_pb:
            offsprings[i] = toolbox.mutate(offsprings[i])
            del offsprings[i].fitness.values

    return offsprings


def run_GA(pop_size, gen_num=100, cross_pb=0.7, mutation_pb=0.3, verbose=False):
    def evaluate_population(population, gen_idx):
        # Evaluate the individuals with an invalid fitness
        invalid_inds = [ind for ind in population if not ind.fitness.valid]
        finesses = toolbox.map(toolbox.evaluate, invalid_inds)
        for ind, fit in zip(invalid_inds, finesses):
            ind.fitness.values = (fit,)

        # Record generation
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen_idx, nevals=len(invalid_inds), **record)
        if verbose:
            print(logbook.stream)

    logbook = tools.Logbook()
    times = []
    last_time = datetime.now()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialize population
    pop = toolbox.population(n=pop_size)

    evaluate_population(pop, 0)

    # Start Generational Loop
    for gen in range(1, gen_num + 1):
        # Record Generation Time
        cur_time = datetime.now()
        times.append((gen, (cur_time - last_time).total_seconds()))
        last_time = cur_time

        # Parental Selection
        parents = toolbox.select(pop, pop_size)

        offsprings = create_offsprings(parents, cross_pb, mutation_pb)

        # Evaluate the individuals with an invalid fitness
        evaluate_population(offsprings, gen)

        # Replace population
        pop[:] = offsprings

        fitness_values = [ind.fitness.values for ind in pop]
        max_fitness = max(fitness_values)
        # dump_population(gen, pop, max_fitness, fitness_values)

    return pop, logbook, times


# Main
if __name__ == '__main__':
    pop_size = 100
    init_GA()

    population, logbook, times = run_GA(pop_size, gen_num=500, verbose=True, mutation_pb=0.6)
