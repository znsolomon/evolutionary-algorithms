import numpy as np
import NSGA3
import supportFunctions as sup
from generateData import get_sample
from matplotlib import pyplot as plt


class Solution:
    def __init__(self, x, c):
        self.C = c  # Matrix of coordinator proportions (boolean)[n, m]
        self.X = x  # Matrix of teaching proportions (float)[n, m]

    def compare_to(self, sol):  # Compares to another Solution object
        if np.array_equal(self.X, sol.X) and np.array_equal(self.C, sol.C):
            return True
        else:
            return False


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
                self_gen = get_sample(alpha=0.1, mnb=36, mxb=20, penalties=penalties)
                dimensions = 7  # M
                boundary = i+1  # p
                inside = j+1

                try:
                    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
                        NSGA3.NSGA3(200, sup.cost, sup.crossover, sup.swap_mutation, sup.create_random,
                        initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                        data=self_gen, pop_size=200, passive_archive=1)
                    test_hv[k] = stats.hv[-1]  # Get last value of hypervolume (most optimised)
                except np.linalg.LinAlgError:
                    k -= 1
            struc_hv[i, j] = np.mean(test_hv)
    return struc_hv


def adjustable_structure_points(boundary, inside):
    penalties = np.array([0.1, 0.1, 0.1])
    self_gen = get_sample(alpha=0.1, mnb=36, mxb=20, penalties=penalties)

    dimensions = 7  # M
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        NSGA3.NSGA3(200, sup.cost, sup.crossover, sup.swap_mutation, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=self_gen, pop_size=200, passive_archive=1)
    return stats


def random_pop_compare():



def basic_nsga3():
    penalties = np.array([0.1, 0.1, 0.1])
    self_gen = get_sample(alpha=0.1, mnb=36, mxb=20, penalties=penalties)

    # Previously used: boundary = 10, inside = 4)
    dimensions = 7  # M
    boundary = 3  # p
    inside = 2
    # Gives 84 reference points on the boundary and 36 on the inside, total of 120
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        NSGA3.NSGA3(200, sup.cost, sup.crossover, sup.swap_mutation, sup.create_random,
                    initial_population=[], boundary_p=boundary, inside_p=inside, M=dimensions,
                    data=self_gen, pop_size=200, passive_archive=1)


if __name__ == '__main__':
    struc_hv = np.zeros((5, 1))
    for i in range(5):
        struc_hv[i] = adjustable_structure_points(4, 5).hv[-1]
    print(np.mean(struc_hv))
