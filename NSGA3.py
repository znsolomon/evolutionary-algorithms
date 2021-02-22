import numpy as np

from recursive_parento_shell_with_duplicates import recursive_pareto_shell_with_duplicates


class Statistics:
    def __init__(self):
        self.prop_non_dom = None
        self.mn = None
        self.hv = None
        self.gen_found = None


def NSGA3(generations, cost_function, crossover_function, mutation_function,
          random_solution_function, initial_population, boundary_p, inside_p, M,
          data, passive_archive, pop_size, extreme_switch=0):
    """
    INPUTS

    generations = number of generations to run optimiser for
    cost_function = handle of function to be optimised
    crossover_function = handle of crossover function
    mutation_function = handle of mutation function
    random_solution_function = handle of function which supplies random legal solutions
    initial_population = holds initial solutions for evaluation, pass an empty matrix,
        [], if no initial solutions available
    boundary_p = number of projection points on simplex boundary (scales up with dimension)
    inside_p = number of projection points on inside boundary (scales up with dimension)
    M = Input dimension of problem (Number of objectives?)
    data = structure holding bounds to be used if preset bounds argument is
        1(l_bound and u_bound vectors for problem dimension)
        and any additional project specific data(e.g.staff / module data for staff allocation optimisation)
    passive_archive = OPTIONAL ARGUMENT(set at 1 if not % provided).
        If equal to 1 a passive archive tracking best solutions evaluated during run is maintained and returned
        in stats structure
    no_obj: number of objectives
    extreme_switch = OPTIONAL ARGUMENT If set at 1 the solutions at the extremes
        (minimising each criterion) are always preserved in the selection from one generation to the next
        Default 1
    preset_bounds = OPTIONAL ARGUMENT if preset_bounds is 1
        then the bounds in data argument are to be used in simplex projection, set at 0 if not provided

    OUTPUTS

    P = Final search population
    Y = Objective values of final serach population
    Zsa = Projection of P
    Pa = Non - dominated subset of P
    Ya = Non - dominated subset of Y
    stats = Structure of various recorded statistics

    REQUIRES

    recursive_pareto_shell_with_duplicates function
    """

    hv_samp_number = 10000
    structure_flag = 0
    P = []
    stats = Statistics()
    start_point = 0
    if initial_population:
        P = initial_population
        start_point = len(P) + 1

    if not passive_archive:
        passive_archive = 1

    # create structured points if aspiration points not passed in
    Zsa = get_structure_points(M, boundary_p, inside_p)

    while pop_size % 4 > 0:
        pop_size = pop_size + 1

    print(f"Population size is: {pop_size}\n")

    for i in range(start_point, pop_size):
        P.append(random_solution_function(data))
    P = np.array(P)

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
        stats.gen_found = np.zeros(Ya.shape[0])  # track which generation a Pareto solution was discovered
        hv_points = np.random.randint(0, 1, size=(hv_samp_number, M))
        hv_points = np.multiply(hv_points, np.tile(data.mxb - data.mnb, (hv_samp_number, 1)))
        hv_points = hv_points + np.tile(data.mnb, (hv_samp_number, 1))
        samps = 0

    for g in range(generations):
        if g % 10 == 0:
            print(f"generation {g}, pop_size {pop_size}, passive archive size {len(Ya)} \n")
            # min(Y)  --> What is this line doing?
        [P, Y, Pa, Ya, non_dom, S, Ry] = evolve(Zsa, P, Y, pop_size, cost_function, crossover_function,
                                                mutation_function, structure_flag, data, Pa, Ya, passive_archive,
                                                extreme_switch)
        if passive_archive:
            stats.prop_non_dom[g] = proportion_nondominated(Y[non_dom, :], Ya)
            stats.mn[g, :] = min(Y)
            [stats.hv[g], hv_points, samps] = est_hv(data.mnb, data.mxb, Ya, hv_points, samps)
            stats.A[g].Y = Y
            stats.A[g].Ya = Ya

            if g % 10 == 0:
                print('Prop dominated %f, MC samples %d, hypervolume %f\n',
                      stats.prop_non_dom[g], samps + hv_samp_number, stats.hv[g])

    return [P, Y, Zsa, Pa, Ya, stats]


def est_hv(mnb, mxb, Ya, hv_points, samps):
    [hv_samp_number, m] = hv_points.shape

    to_remove = np.array([])
    for i in range(hv_points.shape):
        if sum(sum(Ya <= np.tile(hv_points[i, :], (Ya.shape[0], 1), 2) == m)) > 0:
            to_remove = np.append(to_remove, i)

    hv_points[to_remove, :] = []
    removed = len(to_remove)

    # estimate hypervolume
    hv = (hv_samp_number - removed) / (samps + hv_samp_number)

    # update number dominated
    samps = samps + removed

    # refill random samps to 1000
    new_points = np.random.randint(0, 1, size=(removed, m))
    new_points = np.multiply(new_points, np.tile(mxb - mnb, (removed, 1)))
    new_points = new_points + np.tile(mnb, (removed, 1))
    hv_points = np.append(hv_points, new_points)

    return hv, hv_points, samps


def proportion_nondominated(Y, Ya):
    [n, m] = Y.shape
    p = 0
    for i in range(n):
        ge = sum(sum(Ya <= np.tile(Y[i, :], (Ya.shape[0], 1), 2) == m))
        if ge > 0:
            if sum(sum(Ya == np.tile(Y[i, :], (Ya.shape[0], 1), 2) == m)) < ge:  # at least one must dominate
                p = p + 1
    p = (n - p) / n

    return p


def get_structure_points(M, boundary_p, inside_p):
    Zs = get_simplex_samples(M, boundary_p)
    Zs_inside = get_simplex_samples(M, inside_p)
    Zs_inside = Zs_inside / 2  # retract
    Zs_inside = Zs_inside + 0.5 / M  # project inside
    Zs = np.append(Zs, Zs_inside)

    return Zs


def get_simplex_samples(M, p):
    lb = np.linspace(0, p)  # Evenly spaced numbers from o to p
    lb = lb / p
    Zs = []

    for i in range(0, p + 1):  # for lambda in turn
        tmp = np.zeros((M, 1))  # initialise holder for reference point
        tmp = fill_sample(tmp, lb, i, 0, M)
        Zs = np.append(Zs, tmp)

    return Zs


def fill_sample(tmp, lb, lambda_index, layer_processing, M):
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


def nondominated_sort(Ry, extreme_switch):  # Sorts Ry by shell order, ignoring duplicates
    # extreme_switch =1 unless modified
    # Eliminate duplicates from Ry
    Ry = np.unique(Ry, axis=0)
    P_ranks = recursive_pareto_shell_with_duplicates(Ry, extreme_switch)
    # Sort P_ranks into shell order
    m_value = int(max(P_ranks))  # Highest shell number
    shell_values = np.unique(P_ranks)
    P_indexed = {}
    for shell in range(m_value):
        shell_indexes = np.argwhere(P_ranks == shell)
        P_indexed[shell_values[shell]] = shell_indexes

    return P_indexed


def evolve(Zsa, P, Y, N, cost_function, crossover_function, mutation_function,
           structure_flag, data, Pa, Ya, passive_archive, extreme_switch):
    # Za Aspiration points
    # Zr reference points
    # P structure of parents
    # Y objective evaluations of P, matrix | P | by M

    S = []
    Q = crossover_function(P, data)
    Q = mutation_function(Q, data)
    # EVALUATE CHILDREN
    Qy = []  # could preallocate given number of objectives
    for j in range(len(Q)):
        Qy.append(cost_function(Q[j], data))
    Qy = np.array(Qy)

    """if passive_archive:  RE-ENABLE PASSIVE ARCHIVE WHEN THE REST OF THE CODE IS DONE
        P_ranks = recursive_pareto_shell_with_duplicates(Qy, 0)
        to_compare = np.argwhere(P_ranks == 0)
        to_compare = [i[0] for i in to_compare]
        for index in to_compare:
            print(index)
            Ya[index, :] = Qy[index, :]
            Pa[index] = Q[index]"""

    # MERGE POPULATIONS
    R = np.concatenate((P, Q), axis=0)
    Ry = np.concatenate((Y, Qy), axis=0)
    # TRUNCATE POPULATION TO GENERATE PARENTS FOR NEXT GENERATION
    F = nondominated_sort(Ry, extreme_switch)
    # each element of F contains the indices of R of the respective shell
    nd = F[0]

    # Fill S up with shells
    i = 0
    while len(S) < N:
        S = np.append(S, F[i])
        i += 1

    P = []
    Yp = []
    if len(S) != N:  # Concatenate last shell
        indices_used = []
        for j in range(i - 1):
            indices_used = np.append(indices_used, F[j])
            P = np.append(P, R[F[j]])
            Yp = np.append(Yp, Ry[F[j], :])

        Fl = F[i]  # elements of this last shell now need to be chosen
        K = N - len(P)  # specifically K elements
        [Yn, Zr] = normalise(S, Ry, Zsa, structure_flag)

        [index_of_closest, distance_to_closest] = associate(S, Yn, Zr)
        Zr_niche_count = get_niche_count(Zr, index_of_closest(S[0:len(P)]))
        [P, Y, indices_used] = niching(K, Zr_niche_count, index_of_closest, distance_to_closest,
                                       Fl, P, R, Yp, Ry, indices_used)

    else:
        P = R[S]
        Y = Ry[S, :]

    return [P, Y, Pa, Ya, nd, S, Ry]


def normalise(S, Y, Zsa, structure_flag):
    # OUTPUTS
    # Yn = normalised objectives
    # Assumes no preset bounds
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
        scalariser = np.ones((len(S), M))
        scalariser[:, j] = 0
        scalarised = np.sum(np.multiply(Yn[S, :], scalariser), axis=1)
        scalarised[to_remove] = np.inf
        # ensure matrix isn't singular by excluding elements already selected
        for k in range(j):
            ind = scalarising_indices[k]
            scalarised[ind] = np.inf
        i = np.argmin(scalarised)
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


def niching(K, Zr_niche_count, index_of_closest, distance_to_closest, Fl, P, R, Yp, Y, iu):
    # returns indices of final selected population

    k = 1

    while k <= K:
        [j_min] = min(Zr_niche_count)
        I = np.nonzero(Zr_niche_count == j_min)  # get indices of Zr elements which have smallest niche count
        j_bar = np.random.permutation(len(I))
        j_bar = I[j_bar[1]]  # get random index of element of Zr which has lowest niche count
        # get members of Fl which have the j_bar element of Zr as their guide
        Ij_bar = np.nonzero(index_of_closest(Fl) == j_bar)
        if Ij_bar:  # is j_bar index in Fl?
            if Zr_niche_count(j_bar) == 0:  # no associated P member with ref point
                chosen_index = min(distance_to_closest(Fl(Ij_bar)))[1:]  # get index of closest matching member of Fl
            else:
                indices = np.random.permutation(len(Ij_bar))
                chosen_index = indices[1]
            P = [P, R(Fl(Ij_bar[chosen_index]))]  # add to P
            Yp = np.append(Yp, Y[Fl[Ij_bar[chosen_index]], :])
            Zr_niche_count[j_bar] = Zr_niche_count(j_bar) + 1
            iu = np.append(iu, Fl[Ij_bar[chosen_index]])
            Fl[Ij_bar[chosen_index]] = []  # remove from consideration next time
            k = k + 1
        else:
            Zr_niche_count[j_bar] = np.inf
            # put niche count to infinity so it will not be considered in the next loop, same as removing from Zr

    return [P, Yp, iu]


def get_niche_count(Zr, indices):
    # indices =
    niche_count = np.zeros(len(Zr), 1)
    for i in range(len(niche_count)):
        niche_count[i] = sum(indices == i)

    return [niche_count]
