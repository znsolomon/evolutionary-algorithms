import numpy as np
from pymoo.factory import get_performance_indicator

from NSGA3 import Statistics, nondominated_sort, est_hv
from recursive_parento_shell_with_duplicates import recursive_pareto_shell_with_duplicates


def NSGA2(generations, cost_function, crossover_function, mutation_function,
          random_solution_function, initial_population, M, data, passive_archive, pop_size):
    """
    Runs NSGA-2 algorithm on a population.
    :param generations: number of generations to run optimiser for
    :param cost_function: handle of function to be optimised
    :param crossover_function: handle of crossover function
    :param mutation_function: handle of mutation function
    :param random_solution_function: handle of function which supplies random legal solutions
    :param initial_population: holds initial solutions for evaluation, pass an empty matrix,
        [], if no initial solutions available
    :param M: Number of objectives
    :param data: 'Data' object holding information about the problem
    :param passive_archive: If 1, statistics are tracked throughout the generations
    :param pop_size: Size of population
    :return:
    P = Final search population
    Y = Objective values of final serach population
    Pa = Non - dominated subset of P
    Ya = Non - dominated subset of Y
    results = Structure of various recorded statistics
    """

    P = []
    stats = Statistics()
    start_point = 0
    if initial_population:
        P = initial_population
        start_point = len(P) + 1

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
        [P, Y, Pa, Ya, Ry_repeats] = evolve(P, Y, pop_size, cost_function, crossover_function,
                                                mutation_function, data, Pa, Ya, passive_archive)
        if passive_archive:
            stats.prop_non_dom[g] = len(Pa) / len(Y)
            stats.mn[g, :] = np.amin(Y, axis=0)
            stats.hv[g] = est_hv(Y)
            stats.ry_repeats[g] = Ry_repeats

            if g % 10 == 0:
                print(f"Prop non-dominated {stats.prop_non_dom[g]}, "
                      f"hypervolume {stats.hv[g]}\n")

    return [P, Y, Pa, Ya, stats]


def evolve(P, Y, N, cost_function, crossover_function, mutation_function, data, Pa, Ya, passive_archive):
    """
    Evolves the population.
    :param P: Population of solutions
    :param Y: Objective values of population
    :param N: Size of population
    :param cost_function: supportFunctions.py cost()
    :param crossover_function: supportFunctions.py crossover()
    :param mutation_function: supportFunctions.py swap_mutation()
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
        chosen_index = crowding_sort(Ry[Fl.flatten(), :], K)  # Get indices of the selected solutions
        for item in chosen_index:
            P = np.append(P, R[item])
            Yp = np.append(Yp, Ry[[item], :], axis=0)
    else:
        P = R[S.astype(int)]
        Yp = Ry[S.astype(int), :]

    return [P, Yp, Pa, Ya, ry_repeats]


def crowding_sort(Y, k):
    """
    Performs crowding sort on subpopulation
    :param Y: Subpopulation index values
    :param k: Number of solutions to take out of Y
    :return: Index values of chosen solutions
    """
    x, y = Y.shape  # x: no. solutions, y: no. objectives
    crowding_distance_matrix = np.zeros((x, y))
    Yn = (Y - Y.min(0)) / Y.ptp(0)  # Normalise scores (ptp: max-min)
    for j in range(y):
        crowding_distance = np.zeros(x)
        # Boundary scores set to maximum
        crowding_distance[0] = 1
        crowding_distance[-1] = 1
        sort_y = np.sort(Yn[:, j])  # Sorts Yn in order of the chosen objective
        index_y = np.argsort(Yn[:, j])
        for i in range(1, x-1):
            crowding_distance[i] = sort_y[i-1] - sort_y[i+1]
        # Re-sort to original order
        rY_order = np.argsort(index_y)
        sort_crowding_distance = crowding_distance[rY_order]
        crowding_distance_matrix[:, j] = sort_crowding_distance
    crowding_sum = np.sum(crowding_distance_matrix, axis=1)
    select_index = np.argsort(crowding_sum)
    chosen_indicies = []
    for i in range(x):
        if select_index[i] < k:  # Not <= because index value of 0 will be added
            chosen_indicies.append(i)
    return chosen_indicies
