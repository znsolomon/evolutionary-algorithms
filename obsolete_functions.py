def swap_random(P, data):
    """
    generates a random legal solution x

    INPUTS
    data = data used in allocation, in structure.
    data.n the number of staff
    data.m should hold the number of modules.
    data.external_allocation holds the amount of a module which is
         delivered by staff *outside* of the set being allocated to (e.g. from other departments/external speakers).
    data.limited_module_indices holds indices of modules where staff are
     limited on the proportion they should be allocated (e.g. on project modules)
    data.limited_X holds these limits.
    data.increment_number holds the 'chunk' numbers that each module's
        teaching is broken down into equal size chunks of.
    data.duplicated_coord_module_indices{j} holds the jth set of modules
        which codeshare, and therefore should have the same coordinator and
        teaching staff assigned

    OUTPUTS
    x = legal solution
"""

    X = np.zeros(data.m, data.n)
    C = np.zeros(data.m, data.n)
    for i in range(data.m):  # for each module in turn
        rn = np.random.permutation(range(data.n))
        k = np.random.permutation(range(len(rn)))
        rn = rn[k]
        old_rn = rn
        rn = rn[1]
        if sum(data.preallocated_X[i,:]) > 0:
            rn = np.argwhere(data.allocation_mask[i,:]>0)

        if len(rn)>1:
            inc = (data.increment_number[i]-data.external_allocation[i])/len(rn)
            X[i, rn] = inc
        else:
            while (sum(X[i, :])+data.external_allocation[i]) < data.increment_number[i]:
                k = np.random.permutation(range(len(old_rn)))
                index = old_rn[k[0]]
                X[i, index] = X[i, index] + 1

        index = max(X[i, :])
        C[i, index] = 1
    x = Solution(X, C)
    P[0].s = x

    x = P[0].s

    # Error checking?
    if sum(sum(x.X-floor(x.X)))!=0:
        x.X
        print('partial')
    if sum((sum(x.X,2)+data.external_allocation)!=data.increment_number)>0: #  print out if there is an issue
        x.X
        [sum(x.X,2), data.external_allocation, data.increment_number,
         data.module_minimum, sum(data.preallocated_X,2), sum(data.allocation_mask,2)] data.preallocated_X(end,:)
        print('not matching')

    P[1].s = x
    P = teaching_constraints(P, data)  # apply constraints
    x = P[1].s

    return x


def nondominated_sort(Ry, extreme_switch):  # Sorts Ry by shell order, ignoring duplicates
    # extreme_switch =1 unless modified
    F = np.array([])
    [N, M] = Ry.shape
    P_ranks = recursive_pareto_shell_with_duplicates(Ry, extreme_switch)
    raw = P_ranks
    # identify and strip duplicates
    m_value = max(P_ranks) + 1
    # strip out individual minimises to protect
    if extreme_switch:
        I = np.nonzero(P_ranks == 1)
        indices = min(Ry[I, :], [], 1)[1:]
        P_ranks[I[indices]] = 0
    # now remove duplicates
    for i in range(N - 1):
        vec = np.tile(Ry[i, :], (N - i, 1))
        eq_v = vec == Ry[i + 1:, :]
        ind = np.nonzero(sum(eq_v, 2) == M)
        P_ranks[ind + i] = m_value  # move duplicates to worst shell

    for i in range(max(P_ranks)):
        F[i + 1].I = np.nonzero(P_ranks == i)

    return [F, raw]


def update_passive(Pa, Ya, Qy, Q):
    q_shell = recursive_pareto_shell_with_duplicates(Qy, 0)  # gets pareto relationship of Q
    for i in range(len(Qy)):
        if q_shell[i] == 0:  # if not dominated
            Ya[i, :] = Qy[i, :]
            Pa[i] = Q[i]
        Old code:
        if sum(sum(Ya <= np.tile(Qy[i, :], (Ya.shape[0], 1)), 2 == Ya.shape[1])) == 0:  # if not dominated
            indices = sum(Ya >= np.tile(Qy[i, :], (Ya.shape[0], 1), 2)) == Ya.shape[1]
            Ya[indices, :] = []
            Pa[indices] = []
            Ya = np.append(Ya, Qy[i, :])
            Pa = [Pa, Q(i)]

    return [Pa, Ya]


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
    [F, raw] = nondominated_sort(Ry, extreme_switch)
    # each element of F contains the indices of R of the respective shell
    nd = sum(raw == extreme_switch)
    nd = np.linspace(0, min(nd, Y.shape[0]))

    i = 1
    while len(S) < N:
        S = np.append(S, F[i].I)
        i = i + 1

    P = []
    Yp = []
    if len(S) != N:
        indices_used = []
        for j in range(i - 2):
            indices_used = np.append(indices_used, F[j].I)
            P = [P, R[F[j].I]]
            Yp = np.append(Yp, Ry[F[j].I, :])

        Fl = F(i - 1).I  # elements of this last shell now need to be chosen
        K = N - len(P)  # specifically K elements
        [Yn, Zr] = normalise(S, Ry, Zsa, structure_flag, data)

        [index_of_closest, distance_to_closest] = associate(S, Yn, Zr)
        Zr_niche_count = get_niche_count(Zr, index_of_closest(S[0:len(P)]))
        [P, Y, indices_used] = niching(K, Zr_niche_count, index_of_closest, distance_to_closest,
                                       Fl, P, R, Yp, Ry, indices_used)

    else:
        P = R[S]
        Y = Ry[S, :]

    return [P, Y, Pa, Ya, nd, S, Ry]
