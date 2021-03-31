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


if __name__ == '__main__':
    penalties = np.array([0.1, 0.1, 0.1])
    self_gen = get_sample(alpha=0.1, mnb=36, mxb=20, penalties=penalties)

    dimensions = 7  # M
    divisions = 4  # p
    [population, obj_values, struc_points, pop_archive, obj_archive, stats] = \
        NSGA3.NSGA3(200, sup.cost, sup.crossover, sup.swap_mutation, sup.create_random,
                initial_population=[], boundary_p=(dimensions + divisions - 1), inside_p=divisions, M=dimensions,
                data=self_gen, pop_size=200, passive_archive=1)
    plt.figure(1)
    plt.title("Hypervolume over generations")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.plot(range(200), stats.hv)
    plt.show()
    plt.figure(2)
    plt.title("Proportion of crossover/mutated scores repeated over generations")
    plt.xlabel("Generation")
    plt.ylabel("Proportion scores repeated")
    plt.plot(range(200), stats.ry_repeats)
    plt.show()
    plt.figure(3)
    plt.title("Proportion of non-dominated solutions over generations")
    plt.xlabel("Generation")
    plt.ylabel("Proportion non-dominated")
    plt.plot(range(200), stats.prop_non_dom)
    plt.show()
