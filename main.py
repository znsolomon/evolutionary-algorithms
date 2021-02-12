import numpy as np
import NSGA3
import supportFunctions as sup

class Data:
    def __init__(self, n, m, w, h, c, d, p, t, r, pref, term, alpha, pen, mnb, mxb,
                 constraints_on=True, pre_mod_in=None, pre_c=None, pre_x=None, increment_number=0.05):
        # Required variables:
        self.n = n  # NO. staff (int)
        self.m = m  # NO. modules (int)
        self.workload = w  # Non-teaching workloads of each staff member [n]
        self.h = h  # Contractual hours of each staff member [n]
        self.c_matrix = c  # Coordinator hours of each module [m]
        self.d_matrix = d  # Contact hours of each module [m]
        self.p_matrix = p  # Prep time of each module [m]
        self.t = t  # Tij shows if staff i has taught module j [n, m]
        self.r = r  # Rij is percent of module j staff i has taught [n, m]
        self.pref = pref  # Enum of module preferences [n, m]
        self.t_matrix = term   # Enum of which term each module is taught in: 1/2 [m]
        self.alpha = alpha  # How much less prep time a module takes if it has been taught before (int)
        self.mnb = mnb  # (test: mnb = n)???
        self.mxb = mxb  # (test: mxb = m)???
        self.penalties = pen  # Penalty coefficients
        # Optional variables:
        self.constraints_on = constraints_on  # Whether to use constraints (bool)
        self.increment_number = increment_number  # Minimum 'chunk' of teaching allowed
        self.preallocated_module_indices = pre_mod_in  # Incdices of modules with preallocations [m]
        self.preallocated_C = pre_c  # Coordination preallocations [n, m]
        self.preallocated_X = pre_x  # Teaching preallocations [n, m]


class Solution:
    def __init__(self, x, c):
        self.C = c  # Matrix of coordinator proportions (boolean)[n, m]
        self.X = x  # Matrix of teaching proportions (float)[n, m]


if __name__ == '__main__':
    workload = np.zeros(5)
    contract = np.array([20, 25, 30, 45, 35])  # h
    coordinator = np.array([10, 15, 20])
    contact = np.array([20, 20, 40])
    prep = np.array([5, 10, 15])
    coordinator = coordinator[:, np.newaxis]
    contact = contact[:, np.newaxis]
    prep = prep[:, np.newaxis]
    taught = np.array([(0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (1, 0, 0)])
    prop_taught = np.array([(0, 50, 0), (0, 50, 0), (0, 0, 40), (0, 0, 60), (100, 0, 0)])
    pref = np.array([(0, 0, 2), (0, 0, 2), (1, 0, 0), (1, 0, 0), (1, 0, 0)])
    term = np.array([1, 1, 2])
    penalties = np.array([0.1, 0.1, 0.1])

    self_gen = Data(n=5, m=3, w=workload, h=contract, c=coordinator, d=contact, p=prep, t=taught,
                    r=prop_taught, pref=pref, term=term, alpha=0, mnb=5, mxb=3, pen=penalties, pre_mod_in=[])

    dimensions = 7  # M
    divisions = 8  # p
    NSGA3.NSGA3(50, sup.cost, sup.crossover, sup.swap_mutation, sup.swap_random,
          initial_population=[], boundary_p=(dimensions + divisions - 1), inside_p=divisions, M=dimensions,
                data=self_gen, passive_archive=1, extreme_switch=1)
