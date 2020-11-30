from pymoo.model.problem import Problem
from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot


class Data:
    def __init__(self, n, m, module_mask, staff_limited=None, preallocated_module_indicies=None, preallocated_x=None,
                preallocated_c=None, external_allocation=None, limited_module_indicies=None, limited_x=None,
                 increment_no=None):
        self.n = n  # NO. staff
        self.m = m  # NO. modules
        self.module_mask = module_mask  # Max staff allowed in each module
        self.staff_limited = staff_limited  # If staff should be excluded from teaching
        self.preallocated_module_indicies = preallocated_module_indicies  # Indicies of all module with preallocations
        self.preallocated_x = preallocated_x  # Teaching preallocations
        self.preallocated_c = preallocated_c  # Coordinator preallocations
        self.external_allocation = external_allocation  # Proportion of module taught by external teachers
        self.limited_module_indicies = limited_module_indicies
        # Indicies of modules where staff proportions are limited
        self.limited_x = limited_x  # Contains above limits
        self.increment_no = increment_no  # Minimum 'chunk' of teaching allowed


class Solution:
    def __init__(self, X, C):
        self.C = C  # Matrix of coordinator proportions
        self.X = X  # Matrix of teaching proportions


if __name__ == '__main__':
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    pf = get_problem("dtlz1").pareto_front(ref_dirs)
    get_visualization("scatter", angle=(45, 45)).add(pf).show()
