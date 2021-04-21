import numpy as np

from main import Solution


def get_combined_workload(X, C, w_star, c_matrix, d_matrix, p_matrix, alpha, T):
    """
    Calculates the total workload for each staff member
    :param X: Matrix of teaching proportions
    :param C: Matrix of module co-ordinations
    :param w_star: Array containing non-teaching workload of each staff member
    :param c_matrix: Coordinator hours of each module
    :param d_matrix: Contact hours of each module
    :param p_matrix: Prep hours of each module
    :param alpha: Hyperparameter affecting how much teaching the module already affects prep hours
    :param T: Matrix showing if staff member i has taught module j
    :return: Array containing teaching workloads for each staff member
    """
    # Calculate matrix of teaching loads:
    temp = np.multiply(c_matrix, C)\
           + np.multiply((d_matrix + np.multiply((1 + alpha * T), p_matrix)), X)
    w = (w_star+sum(temp))  # add loads to each staff member
    return w


def unbalanced_workload(w, h):
    """
    Calculates the largest difference between contractual hours and workload
    :param w: Array containing workloads of each teacher
    :param h: Array containing contractual hours of each teacher
    :return: Largest difference between w and h
    """
    return max(np.divide(w, h)) - min(np.divide(w, h))


def staff_total_dissatisfaction(x, p, level):
    """
    Calculates dissatisfaction, scaled by proportion of dissatisfying modules being taught
    :param x: Matrix of which staff are teaching which modules
    :param p: Matrix of staff module preference
    :param level: Amount of dissatisfaction to measure
    :return: The value of X where dissatisfaction is highest
    """
    dissatisfaction = []
    for j in range(x.shape[1]):  # For each teacher
        dis_inc = 0
        for i in range(x.shape[0]):  # For each module
            if p[i, j] >= level:
                dis_inc += p[i, j] * x[i, j]
        dissatisfaction.append(dis_inc)
    return sum(dissatisfaction)


def average_staff_per_module(x):
    """
    Calculates average number of staff teaching each module
    :param x: Matrix of which staff are teaching which modules
    :return: Average number of staff teaching each module
    """
    sum_staff = 0
    for i in range(x.shape[0]):  # For each module
        sum_staff += sum(x[i, :] != 0)
    return sum_staff / x.shape[1]


def peak_load(X, C, h, c_matrix, d_matrix, p_matrix, t_matrix, alpha, T):
    """
    Calculates how each teacher's workload changes between term 1 and term 2, prioritising balanced workloads
    :param X: Matrix of teaching proportions
    :param C: Matrix of teaching coordinations
    :param h: Contractual hours of each staff member
    :param c_matrix: Coordinator hours of each module
    :param d_matrix: Contact hours of each module
    :param p_matrix: Prep hours of each module
    :param t_matrix: Enum of which term each module is taught in (1/2)
    :param alpha: Hyperparameter affecting how much teaching the module already affects prep hours
    :param T: Matrix showing if staff member i has taught module j
    :return: The maximum unbalanced workload (most hours in either term 1 or term 2)
    """
    # Calculate matrix of teaching loads:
    temp = np.multiply(c_matrix, C) \
           + np.multiply((d_matrix + np.multiply((1 + alpha * T.reshape(X.shape)), p_matrix)), X)
    t_matrix = t_matrix[:, np.newaxis]
    return max(abs(sum(np.divide(np.multiply(temp, (t_matrix == 1)) - np.multiply(temp, (t_matrix == 2)), h))))


def variation_from_previous_year_teach(X, X_old):
    """
    Calculates how this year's teaching differs from last year's
    :param X: This year's teaching allocation
    :param X_old: Last year's teaching allocation
    :return: Collected difference between X and X_old
    """
    return sum(sum(abs(X - X_old.reshape(X.shape))))


def cost(s, data):
    """
    Finds the objective values (7) of a solution
    :param s: The solution
    :param data: Data object containing information about the problem
    :return: Objective vector of s
    """
    X = s.X / np.tile(data.increment_number, (1, data.n))
    data.pref = np.reshape(data.pref, X.shape)  # Fit preferences to shape of X
    y = np.zeros(7)
    w = get_combined_workload(X, s.C, data.workload, data.c_matrix, data.d_matrix, data.p_matrix, data.alpha, data.t)
    y[0] = sum(w) / sum(data.h)
    y[1] = unbalanced_workload(w, data.h)
    y[2] = staff_total_dissatisfaction(X, data.pref, 1)
    y[3] = staff_total_dissatisfaction(X, data.pref, 2)
    y[4] = average_staff_per_module(X)
    y[5] = peak_load(X, s.C, data.h, data.c_matrix, data.d_matrix, data.p_matrix, data.t_matrix, data.alpha, data.t)
    y[6] = variation_from_previous_year_teach(X, data.r / 100)

    if data.constraints_on:
        for i in range(data.m):  # For each module
            if sum(s.C[i, :]) != 1:  # Total co-ordination value of each module must equal 1
                y = y + (data.penalties[0] / abs(sum(s.C[i, :]) - 1))  # Penalise
            if sum(s.X[i, :]) != 1:  # Whole module must be taught
                y = y + (data.penalties[1] * abs(sum(s.X[i, :]) - 1))  # Penalise
            for j in range(data.n):  # For each staff member
                if s.X[i, j] < 0:   # Staff cannot teach less than 0 or more than 100% of a module
                    y = y + (data.penalties[2] * abs(s.X[i, j]))
                elif s.X[i, j] > 1:
                    y = y + (data.penalties[2] * abs(s.X[i, j]) - 1)

    return y


def crossover(P, data):
    """
    Performs crossover mutation on population
    :param P: Population of solutions
    :param data: Data object containing information about the problem
    :return: P after crossover has been performed
    """
    k = len(P)
    R_comb = np.random.permutation(k)  # Scramble the order of the population
    for i in range(0, k-1, 2):
        # Each population member gets assigned as a parent once
        parent1 = P[R_comb[i]]
        parent2 = P[R_comb[i+1]]
        mask_x = np.random.randint(data.m)
        mask_y = np.random.randint(data.n)
        child1 = parent1
        child2 = parent2
        if np.random.rand() < 0.8 % 80:  # chance of crossover
            # Swap single array value
            child1.X[mask_x, mask_y] = parent2.X[mask_x, mask_y]
            child1.C[mask_x, mask_y] = parent2.C[mask_x, mask_y]

            child2.X[mask_x, mask_y] = parent1.X[mask_x, mask_y]
            child2.C[mask_x, mask_y] = parent1.C[mask_x, mask_y]

            P[R_comb[i]] = child1
            P[R_comb[i+1]] = child2
    return P


def swap_mutation(P, data):
    """
    Performs swap mutation on population
    :param P: Population of solutions
    :param data: Data object containing information about the problem
    :return: P after mutation has been performed
    """
    # Performs swap mutation on population P
    max_to_vary = 1  # NO. elements to switch on
    for i in range(len(P)):
        for k in range(max_to_vary):
            child = P[i]
            rm = np.random.randint(data.m)  # get a module at random
            I = np.argwhere(child.X[rm, :] > 0)  # Get indices where teaching is happening
            if len(I) != 0:  # some delivery internally
                r = np.random.permutation(len(I))
                I = I[r]  # Randomly permute
                if np.random.rand() < 0.5:
                    child.X[rm, I[0]] = child.X[rm, I[0]] - 1
                    rn = np.random.permutation(data.n)  # Allocate to a random other
                    child.X[rm, rn[0]] = child.X[rm, rn[0]] + 1
                else:  # randomly remove teaching of module from one member of staff and give to another
                    rn = np.random.permutation(data.n)  # allocate to a random other
                    if rn[0] == I[0]:
                        rn = rn[1]
                    else:
                        rn = rn[0]
                    child.X[rm, rn] = child.X[rm, rn] + child.X[rm, I[0]]
                    child.X[rm, I] = 0
                # Always assign coordination to staff teaching most of module
                child.C[rm, :] = 0
                index = np.argmax(child.X[rm, :])
                child.C[rm, index] = 1
            else:  # where no teaching due to external delivery swap coordinator
                child.C[rm, :] = 0
                index = np.random.permutation(data.n)
                child.C[rm, index[0]] = 1
            P[i] = child
    P = teaching_constraints(P, data)
    return P


def create_random(data):
    """
    Creates random solutions, given the data of the problem
    :param data: Data object containing information about the problem
    :return: Solution object
    """
    X = np.zeros((data.m, data.n))
    C = np.zeros((data.m, data.n))
    # Assign coordinations randomly
    for x, y in np.ndindex(C.shape):
        C[x, y] = np.random.randint(0, 1)
    # Assign teaching proportions randomly
    for x, y in np.ndindex(X.shape):
        X[x, y] = np.random.rand()

    x = Solution(X, C)
    return x


def teaching_constraints(P, data):
    for i in range(len(P)):
        s = P[i]
        # ensure preallocations
        if data.preallocated_module_indices:
            for j in data.preallocated_module_indices:
                if sum(data.preallocated_C[j, :]) > 0:
                    if sum(abs(s.C[j, :] - data.preallocated_C[j, :])) != 0:  # some coordinators not matched
                        s.C[j, :] = data.preallocated_C[j, :]

                I = np.argwhere((s.X[j, :] - data.preallocated_X[j, :]) < 0)  # identify where < min teaching load
                if I:  # some minimum teaching not matched
                    s.X[j, I] = data.preallocated_X[j, I]
                    total_load = sum(s.X[j, :]) + data.external_allocation[j]
                    while total_load > (data.increment_number[j]):
                        live = np.argwhere(s.X[j, :] > 0)
                        k = np.random.permutation(len(live))
                        if s.X[j, live[k[1]]] > data.preallocated_X[j, live[k[1]]]:
                            s.X[j, live[k[1]]] = s.X[j, live[k[1]]] - 1
                        total_load = sum(s.X[j, :]) + data.external_allocation[j]
        P[i] = s
    return P
