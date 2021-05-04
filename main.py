import numpy as np

import MOEAD
import NSGA2
import NSGA3
import supportFunctions as sup
from generateData import get_sample
from matplotlib import pyplot as plt


class Solution:
    def __init__(self, x, c):
        self.C = c  # Matrix of coordinator proportions (boolean)[m, n]
        self.X = x  # Matrix of teaching proportions (float)[m, n]

    def compare_to(self, sol):  # Compares to another Solution object
        if np.array_equal(self.X, sol.X) and np.array_equal(self.C, sol.C):
            return True
        else:
            return False

    def remove_teach(self, index):
        self.X = np.delete(self.X, index, axis=1)
        self.C = np.delete(self.C, index, axis=1)


def plot_standard_results(results):
    plt.figure(1)
    plt.title("Hypervolume over generations")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.plot(range(200), results.hv)
    plt.show()
    plt.figure(2)
    plt.title("Proportion of crossover/mutated scores repeated over generations")
    plt.xlabel("Generation")
    plt.ylabel("Proportion scores repeated")
    plt.plot(range(200), results.ry_repeats)
    plt.show()
    plt.figure(3)
    plt.title("Proportion of non-dominated solutions over generations")
    plt.xlabel("Generation")
    plt.ylabel("Proportion non-dominated")
    plt.plot(range(200), results.prop_non_dom)
    plt.show()

    objective_names = ["Total workload", "Balanced workload", "Dissatisfaction 1", "Dissatisfaction 2",
                       "Average staff per module", "Peak load", "Variation from previous year"]
    values = []
    for i in range(7):
        values.append(results.mn[:, i])
    plt.figure(4)
    plt.title("Minimum value of each objective over generations")
    plt.xlabel("Generation")
    plt.ylabel("Minimum objective value")
    for i in range(7):
        plt.plot(range(200), values[i], label=objective_names[i])
    plt.legend()
    plt.show()
    plt.figure(5)
    plt.title("Minimum value of each objective over generations, bounded")
    plt.xlabel("Generation")
    plt.ylabel("Minimum objective value")
    for i in range(7):
        plt.plot(range(200), values[i], label=objective_names[i])
    plt.ylim(0, 100000)
    plt.legend()
    plt.show()


def structured_points_analysis():
    penalties = np.array([0.1, 0.1, 0.1])
    struc_hv = np.zeros((5, 5))
    for i in range(5):  # Value of boundary_p (+1)
        for j in range(5):  # Value of inside_p (+1)
            test_hv = np.zeros((5, 1))
            for k in range(5):  # Repeated experiments to avoid randomness
                # Generate different population each time
                self_gen = get_sample(alpha=0.1, penalties=penalties)
                dimensions = 7  # M
                boundary = i+1  # p
                inside = j+1

                try:
                    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
                        NSGA3.NSGA3(200, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                        initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                        data=self_gen, pop_size=200, passive_archive=1)
                    test_hv[k] = stats.hv[-1]  # Get last value of hypervolume (most optimised)
                except np.linalg.LinAlgError:
                    k -= 1
            struc_hv[i, j] = np.mean(test_hv)
    return struc_hv


def adjustable_structure_points(boundary, inside):
    penalties = np.array([0.1, 0.1, 0.1])
    self_gen = get_sample(alpha=0.1, penalties=penalties)

    dimensions = 7  # M
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        NSGA3.NSGA3(200, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=self_gen, pop_size=200, passive_archive=1)
    return stats


def random_pop_compare(gens, pop_size):
    """
    Creates a random population and compares its hypervolume to the hypervolume of a population
    optimised by NSGA3
    :param pop_size: Size of the population
    :return:
    """
    # Generate data
    penalties = np.array([0.1, 0.1, 0.1])
    self_gen = get_sample(alpha=0.1, penalties=penalties)
    # Generate random population and evaluate their objective values
    rand_P = []
    rand_Y = []
    for i in range(pop_size):
        new_pop = sup.create_random(self_gen)
        rand_P.append(new_pop)
        rand_Y.append(sup.cost(new_pop, self_gen))
    rand_P = np.array(rand_P)
    rand_Y = np.array(rand_Y)
    # Generate hypervolume of random population
    rand_hv = NSGA3.est_hv(rand_Y)
    # Perform NSGA3 on another population
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        basic_nsga3(gens, pop_size, data=self_gen)
    plt.figure(1)
    plt.title("Hypervolume over time of random and optimised population")
    plt.xlabel("Generations")
    plt.ylabel("Hypervolume")
    plt.plot(range(gens), np.tile(rand_hv, gens), label="Random population")
    plt.plot(range(gens), stats.hv, label="Optimised population")
    plt.yticks(np.arange(stats.hv[0], stats.hv[-1]+0.1, 0.1))
    plt.legend()
    plt.show()


def mutation_test(generations, red_gens, pop_size, data=None):
    """
    Test the algorithm on two different mutation operators
    :param generations: Number of generations
    :param red_gens: Number of generations for reduced operators that cannot run the full number
    :param pop_size: Size of population
    :param data: Data relating to the problem
    :return: Statistics from the NSGA3 process for two mutation operators
    """
    penalties = np.array([0.1, 0.1, 0.1])
    if not data:
        data = get_sample(alpha=0.1, penalties=penalties)
    dimensions = 7
    boundary = 3
    inside = 2
    # Gaussian mutation is only stable in half as many generations as the fieldsend method
    [population, obj_values, struc_points, pop_archive, obj_archive, gaus_stats] = \
        NSGA3.NSGA3(red_gens, sup.cost, sup.crossover, sup.mutation_gaussian, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=data, pop_size=pop_size, passive_archive=1)
    [population, obj_values, struc_points, pop_archive, obj_archive, reg_stats] = \
        NSGA3.NSGA3(generations, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=data, pop_size=pop_size, passive_archive=1)
    return reg_stats, gaus_stats


def ea_test(generations, pop_size):
    """
    Compares hypervolume of NSGA2, MOEA/D, and NSGA3, with random search as a baseline
    :param generations: Number of generations
    :param pop_size: Population size
    :return:
    """
    penalties = np.array([0.1, 0.1, 0.1])
    data = get_sample(alpha=0.1, penalties=penalties)
    # Generate random population and evaluate their objective values
    rand_hv = np.zeros(generations)
    print("Random search")
    for i in range(generations):
        print(i)
        rand_P = []
        rand_Y = []
        for j in range(pop_size):
            new_pop = sup.create_random(data)
            rand_P.append(new_pop)
            rand_Y.append(sup.cost(new_pop, data))
        rand_P = np.array(rand_P)
        rand_Y = np.array(rand_Y)
        # Get hypervolume for that generation
        rand_hv[i] = NSGA3.est_hv(rand_Y)

    nsga2 = basic_nsga2(generations, pop_size, data=data)
    nsga3 = basic_nsga3(generations, pop_size, data=data)
    moead = basic_moead(generations, pop_size, data=data)
    plt.figure(1)
    plt.title("Hypervolume over generations for each algorithm")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.plot(range(generations), rand_hv, label="Random search")
    plt.plot(range(generations), nsga2.hv, label="NSGA-II")
    plt.plot(range(generations), moead.hv, label="MOEA/D")
    plt.plot(range(generations), nsga3.hv, label="NSGA-III")
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.title("Proportion of non-dominated solutions over generations for each algorithm")
    plt.xlabel("Generation")
    plt.ylabel("Proportion non-dominated")
    plt.plot(range(generations), nsga2.prop_non_dom, label="NSGA-II")
    plt.plot(range(generations), moead.prop_non_dom, label="MOEA/D")
    plt.plot(range(generations), nsga3.prop_non_dom, label="NSGA-III")
    plt.legend()
    plt.show()


def teach_size(generations, pop_size, sizes, specialists, seniors):
    """
    Tests a variety of sizes for teaching staff
    :param generations: Number of generations
    :param pop_size: Size of population
    :param sizes: Array containing each size of teaching staff
    :param specialists: Array containing each number of specialists
    :param seniors: Array containing each number of seniors
    :return:
    """
    penalties = np.array([0.1, 0.1, 0.1])
    results = []  # Stores stats of each test
    for i in range(len(sizes)):
        data = get_sample(alpha=0.1, penalties=penalties, lecturers=sizes[i], spec=specialists[i], res=seniors[i])
        results.append(basic_nsga3(generations, pop_size, data=data))
    plt.figure(1)
    plt.title("Hypervolume over generations for each number of lecturers")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    for i in range(len(sizes)):
        label = str(sizes[i])
        hv = results[i].hv
        plt.plot(range(generations), hv, label=label)
    plt.legend()
    plt.show()


def dynamic_size(generations, pop_size, sizes, boundary=3, inside=2):
    """
    Dynamically changes the size of teaching staff during the NSGA-III evolution
    :param generations: Number of generations to run each size for
    :param pop_size: Size of population
    :param sizes: Array containing different sizes of staff to use, in order
    :param boundary: Number of divisions on the boundary
    :param inside: Number of divisions on the inside
    :return:
    """
    # Generate initial data
    data = get_sample(alpha=0.1, penalties=np.array([0.1, 0.1, 0.1]), lecturers=sizes[0])
    dimensions = 7  # M
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        NSGA3.NSGA3(generations, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=data, pop_size=pop_size, passive_archive=1)
    for i in range(1, len(sizes)):
        old_size = sizes[i-1]
        new_size = sizes[i]
        # Check if removing or adding new staff
        diff = old_size - new_size
        if diff > 0:  # If removing
            to_remove = np.random.choice(old_size, diff, replace=False)  # Determine index values of staff to be removed
            to_remove = np.sort(to_remove)[::-1]
            print(to_remove)
            for j in to_remove:  # Remove staff member for each value in Data
                data.n = data.n - 1
                data.workload = np.delete(data.workload, j)
                data.h = np.delete(data.h, j)
                data.t = np.delete(data.t, j, axis=1)
                data.r = np.delete(data.r, j, axis=1)
                data.pref = np.delete(data.pref, j, axis=1)
                for k in range(len(population)):  # Remove staff member for each population member
                    try:
                        population[k].remove_teach(j)
                    except IndexError:
                        print("Error")
        # Run NSGA-III again for the new population and data parameters
        [population, obj_values, struc_points, pop_archive, obj_archive, new_stats] = \
            NSGA3.NSGA3(generations, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                        initial_population=population, boundary_p=boundary, inside_p=inside, M=dimensions,
                        data=data, pop_size=pop_size, passive_archive=1)
        stats = stats.add_stats(new_stats)  # Update stats
    plot_standard_results(stats)


def basic_nsga3(generations, pop_size, data=None):
    penalties = np.array([0.1, 0.1, 0.1])
    if not data:
        data = get_sample(alpha=0.1, penalties=penalties)

    # Previously used: boundary = 10, inside = 4)
    dimensions = 7  # M
    boundary = 3  # p
    inside = 2
    # Gives 84 reference points on the boundary and 36 on the inside, total of 120
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        NSGA3.NSGA3(generations, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=data, pop_size=pop_size, passive_archive=1)
    return stats


def basic_nsga2(generations, pop_size, data=None):
    penalties = np.array([0.1, 0.1, 0.1])
    if not data:
        data = get_sample(alpha=0.1, penalties=penalties)

    dimensions = 7  # M
    [population, obj_values, pop_archive, obj_archive, stats] = \
        NSGA2.NSGA2(generations, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                    initial_population=[], M=dimensions, data=data, pop_size=pop_size, passive_archive=1)
    return stats


def basic_moead(generations, pop_size, data=None):
    penalties = np.array([0.1, 0.1, 0.1])
    if not data:
        data = get_sample(alpha=0.1, penalties=penalties)

    dimensions = 7  # M
    boundary = 3  # p
    inside = 2
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        MOEAD.MOEAD(generations, sup.cost, sup.crossover, sup.mutation_fieldsend, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=data, pop_size=pop_size, passive_archive=1)
    return stats


if __name__ == '__main__':
    dynamic_size(200, 200, [36, 34])
