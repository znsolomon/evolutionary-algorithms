class Data:
    def __init__(self, n, m, w, h, c, d, p, t, r, pref, alpha, pen, mnb, mxb,
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
        self.r = r  # Rij is proportion of module j staff i has taught [n, m]
        self.pref = pref  # Enum of module preferences [n, m]
        self.alpha = alpha  # ???
        self.mnb = mnb  # ???
        self.mxb = mxb  # ???
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
    print("Hello world")
