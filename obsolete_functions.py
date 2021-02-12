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
