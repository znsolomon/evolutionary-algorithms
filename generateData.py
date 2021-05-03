import numpy as np


class Data:
    def __init__(self, n, m, w, h, c, d, p, t, r, pref, term, alpha, pen,
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
        self.penalties = pen  # Penalty coefficients for each constraint
        # Optional variables:
        self.constraints_on = constraints_on  # Whether to use constraints (bool)
        self.increment_number = increment_number  # Minimum 'chunk' of teaching allowed
        self.preallocated_module_indices = pre_mod_in  # Incdices of modules with preallocations [m]
        self.preallocated_C = pre_c  # Coordination preallocations [n, m]
        self.preallocated_X = pre_x  # Teaching preallocations [n, m]


def get_sample(alpha, penalties, lecturers=36):
    """
    Generates sample data randomly
    :param alpha: Hyperparameter showing how important previously teaching a module is
    :param penalties: Penalty coefficients of the problem's constraints
    :param lecturers: Default value for n
    :return: Data class instance
    """
    n = lecturers  # Number of lecturers
    m = 20  # Number of modules
    hours = 1800  # Average contractual hours of one staff member
    reg_percent = 0.4  # Percentage of hours spent teaching
    res_percent = 0.2  # Percentage of hours spent teaching for staff who have other responsibilities
    no_spec = 15  # Number of specialist staff members
    no_res = 5  # Number of staff with other responsibilities
    coordinator = 30  # Coordinator hours per module
    contact = 60  # Contact hours per module
    prep_low = 50  # Low bound of prep time per module
    prep_high = 151  # High bound of prep time per module
    ind = np.random.choice(range(n), (no_res+no_spec), replace=False)  # Indices of specialist/responsible lectures
    spec_ind = ind[:no_spec]
    res_ind = ind[no_spec:]
    workload = np.zeros(n)  # Non-teaching workload of each staff
    contract_hours = np.full(n, hours)  # Contractual hours of each staff
    for w in range(n):
        if np.any(res_ind == w):
            workload[w] = hours - (hours * res_percent)
        else:
            workload[w] = hours - (hours * reg_percent)
    c = np.full(m, coordinator)
    d = np.full(m, contact)
    p = np.empty(m)
    c = c[:, np.newaxis]
    d = d[:, np.newaxis]
    for mod in range(m):
        p[mod] = np.random.randint(prep_low, prep_high)
    p = p[:, np.newaxis]
    t = np.empty((m, n), dtype=bool)  # If staff n has taught module m
    r = np.zeros((m, n))  # Proportion of module m that staff n taught last year
    pref = np.zeros((m, n))  # Preference of staff n to module m
    """ Staff logic for t, r and pref:
    Most staff are regular. Regular staff pick 1/4 modules out of the total that to be taught previously, 
    teaching 50% of those modules last year, and rating them at 0. 
    They have a 33% chance to rate other modules either a 0, 1 or 2.
    
    Some staff (no_spec) are specialists. Specialist staff pick 1/5 modules out of the total to be taught previously, 
    teaching 100% of 1/2 of those modules last year, and rating them at 0. 
    They have a 50% chance to rate other modules either a 1 or 2.
    """
    for i in range(n):
        if np.any(spec_ind == i):  # Staff is specialist
            modules_taught = np.random.randint(m, size=int(np.round(m/5)))
            last_year_ind = np.random.randint(len(modules_taught), size=int(np.round((m/5)/2)))
            last_year = modules_taught[last_year_ind]
            for j in range(m):
                if np.any(modules_taught == j):
                    t[j, i] = True
                    pref[j, i] = 0
                    if np.any(last_year == j):
                        r[j, i] = 1
                    else:
                        r[j, i] = 0
                else:
                    t[j, i] = False
                    r[j, i] = 0
                    if np.random.rand() < 0.5:
                        pref[j, i] = 1
                    else:
                        pref[j, i] = 2
        else:  # Staff is regular
            modules_taught = np.random.randint(m, size=int(np.round(m / 4)))
            for j in range(m):
                if np.any(modules_taught == j):
                    t[j, i] = True
                    pref[j, i] = 0
                    r[j, i] = 0.5
                else:
                    t[j, i] = False
                    r[j, i] = 0
                    x = np.random.rand()
                    if x < 0.3:
                        pref[j, i] = 2
                    elif x < 0.6:
                        pref[j, i] = 1
                    else:
                        pref[j, i] = 0
    t_one = np.ones(10)
    t_two = np.ones(10) + 1
    terms = np.append(t_one, t_two)  # Which term modules are in
    self_gen = Data(n, m, workload, contract_hours, c, d, p, t, r, pref, terms, alpha, penalties)
    return self_gen
