import numpy as np
import NSGA3
import supportFunctions as sup
from generateData import get_sample


class Solution:
    def __init__(self, x, c):
        self.C = c  # Matrix of coordinator proportions (boolean)[n, m]
        self.X = x  # Matrix of teaching proportions (float)[n, m]


if __name__ == '__main__':
    penalties = np.array([0.1, 0.1, 0.1])
    self_gen = get_sample(alpha=0.1, mnb=36, mxb=20, penalties=penalties)

    dimensions = 7  # M
    divisions = 4  # p
    NSGA3.NSGA3(50, sup.cost, sup.crossover, sup.swap_mutation, sup.swap_random,
          initial_population=[], boundary_p=(dimensions + divisions - 1), inside_p=divisions, M=dimensions,
                data=self_gen, pop_size=200, passive_archive=1)
