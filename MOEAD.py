import numpy as np

from recursive_parento_shell_with_duplicates import recursive_pareto_shell_with_duplicates
from NSGA3 import perpendicular_distance, est_hv, get_structure_points


class Statistics:
    def __init__(self):
        self.prop_non_dom = None  # Proportion of nondominated solutions each generation
        self.mn = None  # Minimum value of Y (objective values) each generation
        self.hv = None  # Hypervolume indicator each generation
        self.ry_repeats = None  # Proportion of Ry population that aren't unique


def MOEAD(generations, cost_function, crossover_function, mutation_function,
          random_solution_function, initial_population, boundary_p, inside_p, M,
          data, passive_archive, pop_size):
    """
    Runs MOEA/D algorithm on a population.
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

            if g % 10 == 0:
                print(f"Prop non-dominated {stats.prop_non_dom[g]}, "
                      f"hypervolume {stats.hv[g]}\n")

    return [P, Y, Zsa, Pa, Ya, stats]


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

        [index_of_closest, distance_to_closest] = associate(S, Ry, Zsa)

        # Get 'niche count': count of how many individuals are associated with each reference point
        s_targets = S[0:len(P)]
        ioc_targets = []
        for t in s_targets:
            t = int(t)
            ioc_targets.append(index_of_closest[t])
        Zsa_niche_count = np.zeros(len(Zsa))
        for i in range(len(Zsa_niche_count)):
            Zsa_niche_count[i] = np.count_nonzero(ioc_targets == i)

        [P, Y] = niching(K, Zsa_niche_count, index_of_closest, distance_to_closest,
                                       Fl, P, R, Yp, Ry)

    else:
        P = R[S.astype(int)]
        Y = Ry[S.astype(int), :]

    return [P, Y, Pa, Ya, ry_repeats]


def associate(S, Yn, Zr):
    """
    1. Compute reference lines of Zr by joining each reference point with the origin
    2. Get distance from each population member to each reference line
    3. Associate each pop member with the closest reference line
    :param S: indices of population members being considered
    :param Yn: normalised objective vectors of entire population
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
