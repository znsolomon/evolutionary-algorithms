import numpy as np

def get_combined_workload(X, C, w_star, c_matrix, d_matrix, p_matrix, alpha, T):
    # Calculates the total workload for each staff member and puts in an array
    # Calculate matrix of teaching loads:
    temp = np.multiply(c_matrix, C) + np.multiply((d_matrix + np.multiply((1+alpha*T), p_matrix)), X)
    w = (w_star+sum(temp)) # add loads to each staff member
    return w

def unbalanced_workload(w, h):
    return max(np.divide(w, h))- min(np.divide(w, h))

def staff_dissatisfaction(X, P, level, increment_number):
    return max(sum(np.multiply((np.divide(X, np.tile(increment_number, (1, (X, 2).shape)))), (P >= level))))

def staff_total_dissatisfaction(X, P, level, increment_number):
    return sum(sum(np.multiply((np.divide(X, np.tile(increment_number, (1, (X, 2).shape)))), (P >= level))))

def average_staff_per_module(X):
    return sum(sum(X!=0))/(X,1).shape

def peak_load(X, C,  h, c_matrix,d_matrix, p_matrix, t_matrix, alpha, T):
    temp = np.multiply(c_matrix, C) + np.multiply(d_matrix + np.multiply((1 + alpha * T), p_matrix), X)
    return max(abs(sum(np.divide(np.multiply(temp, (t_matrix == 1)) - np.multiply(temp, (t_matrix == 2))), h)))

def variation_from_previous_year_teach(X, X_old, increment_number):
    return sum(sum(abs(X - X_old)))


def cost(s, data):
    """Seven-objective cost function
    INPUTS
    s - solution (matrix of X and C values)
    data - Data structure seen in main.py
    OUTPUTS
    y - objective vector (to minimise)
    W - combined matrix of staff workloads
    """
    X = s.X / np.tile(data.increment_number, (1, data.n))
    y = np.zeros(1, 7)
    w = get_combined_workload(X, s.C, data.w_star, data.c_matrix, data.d_matrix, data.p_matrix, data.alpha, data.T)
    y[1] = sum(w) / sum(data.h)
    y[2] = unbalanced_workload(w, data.h)
    y[3] = staff_total_dissatisfaction(X, data.P, 1, data.increment_number)
    y[4] = staff_dissatisfaction(X, data.P, 2, data.increment_number)
    y[5] = average_staff_per_module(X)
    y[6] = peak_load(X, s.C, data.h, data.c_matrix, data.d_matrix, data.p_matrix, data.t_matrix, data.alpha, data.T)
    y[7] = variation_from_previous_year_teach(X, data.R / 100, data.increment_number)

    if data.constraints_on:
        for i in range(len(data.m)):
            if sum(s.X[i,:] > 0) < data.module_minimum[i]: # Teaching load should be positive
                y[data.objective_mask] = \
                    y[data.objective_mask] + data.mxb * (data.module_minimum[i] - sum(s.X[i,:] > 0))
            # Total teaching allocation of module shouldn't be greater than module maximum
            if sum(s.X[i,:] > data.module_maximum(i)) > 0:
                y[data.objective_mask] = \
                    y[data.objective_mask] + data.mxb * (abs(data.module_maximum[i] - sum(s.X[i,:] > 0)))

        mx_p = 10
        for i in range(len(data.n)):
            if (sum(s.X[data.all_project_indices, i]) > mx_p): # total projects
                y[data.objective_mask] = \
                    y[data.objective_mask] + (sum(s.X[data.all_project_indices, i]) - mx_p) * data.mxb

        # penalise assignment to prevent marked mappings
        temp = sum(sum(s.X[data.prevent == 1]))
        y[data.objective_mask] = y[data.objective_mask]+ temp * data.mxb

    return y


def crossover(P, data):
    # Performs crossover on population P
    k = len(P)
    R_comb = np.random.permutation(k)
    for i in range(0, k-1, 2):
        parent1 = P[R_comb[i]].s
        parent2 = P[R_comb[i+1]].s
        crossover_mask = np.random.rand(data.m, 1) < 0.5
        child1 = parent1
        child2 = parent2
        if np.random.rand() < 0.8 % 80: # chance of crossover
            child1.X[crossover_mask,:] = parent2.X[crossover_mask,:]
            child1.C[crossover_mask,:] = parent2.C[crossover_mask,:]

            child2.X[crossover_mask,:] = parent1.X[crossover_mask,:]
            child2.C[crossover_mask,:] = parent1.C[crossover_mask,:]

            P[R_comb[i]].s = child1
            P[R_comb[i]].s = child2
    return P

def swap_mutation(P, data):
    # Performs swap mutation on population P
    max_to_vary = 1 # NO. elements to switch on
    for i in range(len(P)):
        for k in range(max_to_vary):
            child = P[i].s
            rm = np.random.permutation(data.m) # get a module at random
            rm = rm[1]
            if (data.module_mask(rm) == 1):
                r = np.random.permutation(data.n)
                child.C[rm,:] = 0
                child.X[rm,:] = 0
                child.C[rm, r[1]] = 1 # Swap staff member involved
                child.X[rm, r[1]] = data.increment_number[rm] - data.external_allocation[rm]
            else:
                I = np.argwhere(child.X[rm,:] > 0)  # Get indicides where teaching is happening
                r = np.random.permutation(len(I))
                I = I[r] # Randomly permute
                if I: # some delivery internally
                    if np.random.rand() < 0.5:
                        child.X[rm[1], I[1]] = child.X[rm[1], I[1]] - 1
                        if (len(r) < data.module_mask(rm)): # can add extra staff
                            rn = np.random.permutation(data.n) # Allocate to a random other
                            child.X[rm[1], rn[1]] = child.X[rm[1], rn[1]] + 1
                            # Always assign coordination to staff teaching most of module
                            child.C[rm[1],:] = 0
                            index = max(child.X[rm[1],:])[1:]
                            child.C[rm[1], index] = 1
                        else: # can only shift between staff
                            child.X[rm[1], I[2]] = child.X[rm[1], I[2]] + 1
                            # always assign coordination to staff teaching most of module
                            child.C[rm[1],:] = 0
                            index = max(child.X[rm[1],:])[1:]
                            child.C[rm[1], index] = 1
                    else: # randomly remove teaching of module from one member of staff and give to another
                        rn = np.random.permutation(data.n) # allocate to a random other
                        if rn[1] == I[1]:
                            rn = rn[2]
                        else:
                            rn = rn[1]
                        child.X[rm[1], rn] = child.X[rm[1], rn] + child.X[rm[1], I[1]]
                        child.X[rm[1], I[1]] = 0
                        child.C[rm[1], rn] = max(child.C[rm[1], I[1]], child.C[rm[1], rn])
                        child.C[rm[1], I[1]] = 0
                else: # where no teaching due to external delivery swap coordinator
                    child.C[rm[1],:]=0
                    index = np.random.permutation(data.n)
                    child.C[rm[1], index[1]] = 1
            P[i].s = child
    P = teaching_constraints(P,data)
    return P


def teaching_constraints(P, data):
    for i in range(len(P)):
        s = P[i].s
        # ensure preallocations
        for j in data.preallocated_module_indices:
            if sum(data.preallocated_C[j,:]) > 0:
                if sum(abs(s.C[j,:] - data.preallocated_C[j,:])) != 0: # some coordinators not matched
                    s.C[j,:] = data.preallocated_C[j,:]

            I = np.argwhere((s.X[j,:] - data.preallocated_X[j,:]) < 0) # identify where < min teaching load
            if I: # some minium teaching not matched
                s.X[j, I] = data.preallocated_X[j, I]
                total_load = sum(s.X[j,:]) + data.external_allocation[j]
                while total_load > (data.increment_number[j]):
                    live = np.argwhere(s.X[j,:] > 0)
                    k = np.random.permutation(len(live))
                    if s.X[j, live[k[1]]] > data.preallocated_X[j, live[k[1]]]:
                        s.X[j, live[k[1]]] = s.X[j, live[k[1]]] - 1
                    total_load = sum[s.X[j,:]] + data.external_allocation[j]

        # ensure maximum isn't breached on projects
        for j in data.limited_module_indices:
            I = np.argwhere(s.X[j,:] - data.limited_X[j,:] > 0) # identify where > max teaching load
            if I: # some maximum teaching breached
                s.X[j, I] = data.limited_X[j, I]
                total_load = sum(s.X[j,:])+data.external_allocation[j]
                while total_load < (data.increment_number[j]):
                    live = np.argwhere((s.X[j,:] > 0) + (s.X[j,:] < data.limited_X[j,:]) == 2)
                    if not live:
                        live = np.argwhere(s.X[j,:] < data.limited_X[j,:])
                    k = np.random.permutation(len(live))
                    s.X[j, live[k[1]]] = s.X[j, live[k[1]]] + 1
                    total_load = sum(s.X[j,:]) + data.external_allocation[j]

        # ensure duplicated modules are co-taught
        for j in range(len(data.duplicated_module_indices)):
            I = data.duplicated_module_indices[j]
            s.X[I[2],:] = data.increment_number[I[2]] * s.X[I[1],:] / data.increment_number[I[1]]
            s.C[I[2],:] = s.C[I[1],:]

        for j in range(len(data.duplicated_coord_module_indices)):
            I = data.duplicated_coord_module_indices[j]
            s.C[I[2],:] = s.C[I[1],:]

        # ensure coordinator is a teacher
        for j in data.limited_module_indices:
            I = np.argwhere(s.X[j,:] > 0)
            if I:
                if sum(s.C[j, I]) == 0: # teacher isn't coordinator
                    s.C[j,:] = 0
                    r = np.random.permutation(len(I))
                    s.C[j, I[r[1]]] = 1

        P(i).s = s
    return P
