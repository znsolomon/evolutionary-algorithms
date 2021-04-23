import numpy as np
from pymoo.factory import get_performance_indicator

from recursive_parento_shell_with_duplicates import recursive_pareto_shell_with_duplicates


class Statistics:
    def __init__(self):
        self.prop_non_dom = None  # Proportion of nondominated solutions each generation
        self.mn = None  # Minimum value of Y (objective values) each generation
        self.hv = None  # Hypervolume indicator each generation
        self.ry_repeats = None  # Proportion of Ry population that aren't unique


def NSGA3(generations, cost_function, crossover_function, mutation_function,
          random_solution_function, initial_population, boundary_p, inside_p, M,
          data, passive_archive, pop_size):
    """
    Runs NSGA-3 algorithm on a population.
    :param generations: number of generations to run optimiser for
    :param cost_function: handle of function to be optimised
    :param crossover_function: handle of crossover function
    :param mutation_function: handle of mutation function
    :param random_solution_function: handle of function which supplies random legal solutions
    :param initial_population: holds initial solutions for evaluation, pass an empty matrix,
        [], if no initial solutions available
    :param boundary_p: number of projection points on simplex boundary (scales up with dimension)
    :param inside_p: number of projection points on inside boundary (scales up with dimension)
    :param M: Number of objectives
    :param data: 'Data' object holding information about the problem
    :param passive_archive: If 1, statistics are tracked throughout the generations
    :param pop_size: Size of population
    :return:
    P = Final search population
    Y = Objective values of final serach population
    Zsa = Projection of P
    Pa = Non - dominated subset of P
    Ya = Non - dominated subset of Y
    results = Structure of various recorded statistics
    """

    structure_flag = 0
    P = []
    stats = Statistics()
    start_point = 0
    if initial_population:
        P = initial_population
        start_point = len(P) + 1

    # create structured points if aspiration points not passed in
    Zsa = get_structure_points(M, boundary_p, inside_p)

    while pop_size % 4 > 0:
        pop_size = pop_size + 1

    print(f"Population size is: {pop_size}\n")

    # Create random starting population using the random solution function
    for i in range(start_point, pop_size):
        P.append(random_solution_function(data))
    P = np.array(P)

    # Evaluate costs of the initial population
    Y = []
    for i in range(len(P)):
        Y.append(cost_function(P[i], data))
    Y = np.array(Y)

    Pa = []
    Ya = np.array([])
    if passive_archive == 1:
        P_ranks = recursive_pareto_shell_with_duplicates(Y, 0)
        # P_ranks: pareto-shell rankings of each solution (which shell they are in)
        nondom = np.argwhere(P_ranks == 0)
        nondom = [i[0] for i in nondom]
        Ya = Y[nondom]
        Pa = P[nondom]
        stats.prop_non_dom = np.zeros((generations, 1))
        stats.mn = np.zeros((generations, M))
        stats.hv = np.zeros((generations, 1))
        stats.ry_repeats = np.zeros((generations, 1))

    for g in range(generations):
        print(g)
        if g % 10 == 0:
            print(f"generation {g}, pop_size {pop_size}, passive archive size {len(Ya)} \n")
        [P, Y, Pa, Ya, Ry_repeats] = evolve(Zsa, P, Y, pop_size, cost_function, crossover_function,
                                                mutation_function, structure_flag, data, Pa, Ya, passive_archive)
        if passive_archive:
            stats.prop_non_dom[g] = len(Pa) / len(Y)
            stats.mn[g, :] = np.amin(Y, axis=0)
            stats.hv[g] = est_hv(Y)
            stats.ry_repeats[g] = Ry_repeats
            """ If counting repeats in P and Y, use:
            repeats = []
            for i in range(len(P)):
                if not i in repeats:
                    solution = P[i]
                    for j in range(len(P)):
                        if i != j:
                            compare = P[j]
                            if solution.compare_to(compare):
                                repeats.append(j)
            results.p_repeats[g] = len(repeats) / len(P)
            results.y_repeats[g] = len(np.unique(Y, axis=0)) / len(Y)"""

            if g % 10 == 0:
                print(f"Prop non-dominated {stats.prop_non_dom[g]}, "
                      f"hypervolume {stats.hv[g]}\n")

    return [P, Y, Zsa, Pa, Ya, stats]


def est_hv(Y):
    """
    Gets hypervolume estimate from population
    :param Y: Population objective values
    :return: Hypervolume estimate
    """
    # Find reference point: 1.1 * largest point in each generation
    ref_point = 1.1 * np.amax(Y, axis=0)
    # Find hypervolume given reference point
    hv = get_performance_indicator("hv", ref_point=ref_point)
    return "{:.16f}".format(float(hv.calc(Y)))


def get_structure_points(M, boundary_p, inside_p):
    """
    Place reference points on a normalised hyperplane
    :param M: Number of objectives
    :param boundary_p: Number of divisions for each objective axis (on the boundary)
    :param inside_p: Number of divisions for each objective axis (inside the hyperplane)
    :return: Evenly spaced reference points on a normalised hyperplane
    """
    Zs = get_simplex_samples(M, boundary_p)
    Zs_inside = get_simplex_samples(M, inside_p)
    Zs_inside = Zs_inside / 2  # retract
    Zs_inside = Zs_inside + 0.5 / M  # project inside
    Zs = np.append(Zs, Zs_inside)

    return Zs


def get_simplex_samples(M, p):
    """
    Generate evenly spaced points along a simplex
    :param M: Number of objectives (dimensions)
    :param p: Number of divisions for each objective axis
    :return: Array containing points
    """
    lb = np.linspace(0, p)  # Evenly spaced numbers from 0 to p
    lb = lb / p
    Zs = []

    for i in range(0, p + 1):  # for lambda in turn
        tmp = np.zeros((M, 1))  # initialise holder for reference point
        tmp = fill_sample(tmp, lb, i, 0, M)
        Zs = np.append(Zs, tmp)

    return Zs


def fill_sample(tmp, lb, lambda_index, layer_processing, M):
    """
    Fills a reference point with the correct values
    :param tmp: Reference point to be filled
    :param lb: Evenly spaced numbers from 0 to p
    :param lambda_index: which value of lb is currently being processed
    :param layer_processing: Which layer the recursive algorithm is currently on (starts at 0)
    :param M: Number of objectives
    :return: M-dimensional reference point
    """
    tmp[layer_processing] = lb[lambda_index]  # Sets current layer of tmp to current lambda
    if layer_processing < M - 2:  # For each layer but the second-last and last
        already_used = sum(tmp[0:layer_processing])  # Values of tmp already filled
        # identify valid fillers that can be used:
        valid_indices = np.where(lb <= 1 - already_used + np.finfo(float).eps)
        tmp_processed = np.array([])
        for j in range(len(valid_indices)):
            tmp_new = tmp
            recursive_matrix = fill_sample(tmp_new, lb, j, layer_processing + 1, M)
            tmp_processed = np.append(tmp_processed, recursive_matrix)
    else:  # Second-last (M-1th) layer being processed so last element has to complete sum
        tmp_processed = tmp
        tmp_processed[M-1] = 1 - sum(tmp[0:M - 2])

    return tmp_processed


def nondominated_sort(Ry):
    """
    Sorts Ry by shell order, ignoring duplicates
    :param Ry: Objective values of expanded population (400 members)
    :return: Dictionary of each shell, mapped to index values of each pop member in the shells
    """
    # Eliminate duplicates from Ry
    Ry = np.unique(Ry, axis=0)
    P_ranks = recursive_pareto_shell_with_duplicates(Ry, 0)
    # Sort P_ranks into shell order
    m_value = int(max(P_ranks))  # Highest shell number
    shell_values = np.unique(P_ranks)
    P_indexed = {}
    for shell in range(m_value):
        shell_indexes = np.argwhere(P_ranks == shell)
        P_indexed[shell_values[shell]] = shell_indexes

    return P_indexed


def evolve(Zsa, P, Y, N, cost_function, crossover_function, mutation_function,
           structure_flag, data, Pa, Ya, passive_archive):
    """
    Evolves the population.
    :param Zsa: Structured reference points for use in normalisation
    :param P: Population of solutions
    :param Y: Objective values of population
    :param N: Size of population
    :param cost_function: supportFunctions.py cost()
    :param crossover_function: supportFunctions.py crossover()
    :param mutation_function: supportFunctions.py swap_mutation()
    :param structure_flag:
        Shows if we are using pre-supplied points or structure points created at the algorithm's start
    :param data: Data class instance containing information about the problem
    :param Pa: Current non-dominated set of P
    :param Ya: Objective values of current non-dominated set
    :param passive_archive: Boolean showing if statistics are being recorded
    :return P: Updated P values
    :return Y: Updated Y values
    :return Pa: Updated Pa values
    :return Ya: Updated Ya values
    """
    ry_repeats = None
    S = []
    Q = crossover_function(P, data)
    Q = mutation_function(Q, data)
    # EVALUATE CHILDREN
    Qy = []  # could preallocate given number of objectives
    for j in range(len(Q)):
        Qy.append(cost_function(Q[j], data))
    Qy = np.array(Qy)

    # MERGE POPULATIONS
    R = np.concatenate((P, Q), axis=0)
    Ry = np.concatenate((Y, Qy), axis=0)
    # TRUNCATE POPULATION TO GENERATE PARENTS FOR NEXT GENERATION
    F = nondominated_sort(Ry)
    # each element of F contains the indices of R of the respective shell
    # Save non-dominated set into Pa and Ya for statistics
    nd = F[0]
    nd = [i[0] for i in nd]
    if passive_archive:
        ry_u = np.unique(Ry, axis=0).shape[0]
        ry_repeats = (Ry.shape[0] - ry_u) / Ry.shape[0]
        # print("Proportion scores repeated: " + str((Ry.shape[0] - ry_u) / Ry.shape[0]))
        Pa = R[nd]
        Ya = Ry[nd]

    # Fill S up with shells
    i = 0
    while len(S) < N:
        S = np.append(S, F[i])
        i += 1

    P = []
    Yp = []
    first = True
    if len(S) != N:
        # The next shell is too big to fit in the population for next generation, so it must be ordered
        # First, add the confirmed population for next generation to P:
        for j in range(i - 1):
            P = np.append(P, R[F[j]])
            for item in F[j]:
                if first:
                    first = False
                    Yp = Ry[item, :]
                else:
                    Yp = np.append(Yp, Ry[item, :], axis=0)

        Fl = F[i-1]  # elements of this last shell now need to be chosen
        K = N - len(P)  # specifically K elements
        [Yn, Zr] = normalise(S, Ry, Zsa, structure_flag)

        [index_of_closest, distance_to_closest] = associate(S, Yn, Zr)
        # Get 'niche count': count of how many individuals are associated with each reference point
        s_targets = S[0:len(P)]
        ioc_targets = []
        for t in s_targets:
            t = int(t)
            ioc_targets.append(index_of_closest[t])
        Zr_niche_count = np.zeros(len(Zr))
        for i in range(len(Zr_niche_count)):
            Zr_niche_count[i] = np.count_nonzero(ioc_targets == i)

        [P, Y] = niching(K, Zr_niche_count, index_of_closest, distance_to_closest,
                                       Fl, P, R, Yp, Ry)

    else:
        P = R[S.astype(int)]
        Y = Ry[S.astype(int), :]

    return [P, Y, Pa, Ya, ry_repeats]


def normalise(S, Y, Zsa, structure_flag):
    """
    Normalises objective values, assuming no preset bounds
    :param S: Index values of solutions allowed into the new population
    :param Y: Objective values of entire population
    :param Zsa: Structured reference points
    :param structure_flag:
    :return: Yn = normalised objectives
    """
    S = S.astype(int)
    M = Y.shape[1]  # get number of objectives
    ideal = np.amin(Y, axis=0)  # initialise ideal point by finding smallest value

    Yn = Y - np.tile(ideal, (Y.shape[0], 1))
    """ FROM PAPER:
    Thereafter, the extreme point(zi, max) in each(ith) objective axis is identified
    by finding the solution(x ? St) that makes the corresponding achievement
    scalarizing function(formed with f_i (x) and a weight vector close to ith objective axis) minimum.
    """
    scalarising_indices = []
    to_remove = []
    for j in range(M):  # Find the extreme value of each objective
        # Finds the solutions that minimise the other objectives
        scalariser = np.ones((len(S), M))
        scalariser[:, j] = 0
        scalarised = np.sum(np.multiply(Yn[S, :], scalariser), axis=1)
        scalarised[to_remove] = np.inf
        # ensure matrix isn't singular by excluding elements already selected
        for k in range(j):
            ind = scalarising_indices[k]
            scalarised[ind] = np.inf
        not_found = True
        while not_found:
            i = np.argmin(scalarised)
            if Yn[S[i], j] == 0.0:  # Having extreme values of 0 mean the hyperplane cannot be formed
                scalarised[i] = np.inf  # Therefore they must be discounted
            else:
                not_found = False
        # identify solution along the ith axis
        # (i.e. minimising the other objectives as much as possible)
        scalarising_indices.append(i)
        # Mark any solutions for removal that perform the same as extreme point
        for l in range(S.shape[0]):
            current = Yn[S[l], :]
            if np.array_equal(current, Yn[S[i], :]):
                to_remove.append(l)

    X = Yn[S[scalarising_indices], :]

    a = np.linalg.solve(X, np.ones((M, 1)))  # solve system of linear equations to get weights

    Yn = np.divide(Yn, np.reshape(a, (1, M)))  # rescale

    if structure_flag:
        Zr = Zsa
    else:
        # Zr = np.multiply(Zsa, np.tile(a, (Zsa.shape[0], 1)))
        Zr = np.transpose(np.divide(Zsa, a),)

    return [Yn, Zr]


def perpendicular_distance(direction, point):
    """
    Calculates perpendicular distance between direction and point.
    :param direction:
    :param point:
    :return:
    """
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2))
    return np.sqrt(d)


def associate(S, Yn, Zr):
    """
    1. Compute reference lines of Zr by joining each reference point with the origin
    2. Get distance from each population member to each reference line
    3. Associate each pop member with the closest reference line
    :param S: indices of members
    :param Yn: normalised objective vectors
    :param Zr: reference points
    :return: indexes and distances of population members associated with reference points
    """
    max_s = int(np.amax((S)))
    index_of_closest = np.zeros((max_s+1, 1))
    distance_to_closest = np.zeros((max_s+1, 1))
    for i in range(len(S)):
        D = np.zeros(Zr.shape[0])
        s_index = int(S[i])
        for j in range(Zr.shape[0]):
            D[j] = perpendicular_distance(Yn[s_index], Zr[j])
        closest = np.argmin(D)
        distance_to_closest[s_index] = D[closest]
        index_of_closest[s_index] = closest

    return [index_of_closest, distance_to_closest]


def niching(K, Zr_niche_count, index_of_closest, distance_to_closest, Fl, P, R, Yp, Y):
    """
    1. Identify the set of reference points with minimum niche count (choose one at random if multiples)
    2. If the front has no members associated with reference point, exclude it from further calculation
    3. If niche count = 0, get the closest pop members to the reference point
    4. If niche count >= 1, get one of the associated pop members at random
    5. Add this acquired pop members to the pop for next gen, then repeat until P is full
    :param K: The number of points needed to add to P
    :param Zr_niche_count: Count of how many individuals are associated with each reference point [no. ref points]
    :param index_of_closest: Indices of pop members associated with reference points
    :param distance_to_closest: Distances of pop members associated with reference points
    :param Fl: Shell of pop members needed to look through (index values)
    :param P: Population for next generation
    :param R: Merged full population in current generation
    :param Yp: Objective values of the population for next generation (P)
    :param Y: Objective values of the full population (R)
    :return: Updated P, Yp by adding vales from the next shell
    """
    k = 1
    while k <= K:
        # get indices of Zr elements which have smallest niche count
        I = np.where(Zr_niche_count == Zr_niche_count.min())
        I = np.array(I)
        if I.shape[1] > 1:  # If there's more than one Zr element with smallest niche count
            smallest_niche_ref = np.random.permutation(I.shape[1])
            # get random index of element of Zr which has lowest niche count
            smallest_niche_ref = I[0, smallest_niche_ref[0]]
        else:
            smallest_niche_ref = I
        # get members of Fl which have the smallest_niche_ref element of Zr as their guide
        Ij_bar = []
        for i in range(len(Fl)):
            ind = Fl[i]
            if index_of_closest[ind] == smallest_niche_ref:
                Ij_bar.append(i)
        if Ij_bar:  # is smallest_niche_ref index in Fl?
            if Zr_niche_count[smallest_niche_ref] == 0:  # no associated P member with ref point
                # get index of closest matching member of Fl
                chosen_index = np.argmin(distance_to_closest[Fl[Ij_bar]])
            else:
                indices = np.random.permutation(len(Ij_bar))
                chosen_index = indices[0]
            # Problem: Ij_bar contains values that are bigger than Fl
            P = np.append(P, R[Fl[Ij_bar[chosen_index]]])  # add to P
            Yp = np.append(Yp, Y[Fl[Ij_bar[chosen_index]], :], axis=0)
            Zr_niche_count[smallest_niche_ref] += 1
            Fl[Ij_bar[chosen_index]] = np.empty((1, 1))  # remove from consideration next time
            k = k + 1
        else:
            Zr_niche_count[smallest_niche_ref] = np.inf
            # put niche count to infinity so it will not be considered in the next loop, same as removing from Zr
    return [P, Yp]
