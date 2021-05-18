import numpy as np


def recursive_pareto_shell_with_duplicates(Y, index):
    """
    Recursively computes Pareto shell membership
    :param Y: A n by m matrix of objectives,
        where m is the number of objectives and n is the number of points (pop size)
    :param index: the integer value you wish to attribute to the estimated Pareto-optimal shell (typically 0 or 1)
    :return: n by 1 array of corresponding shell membership values
    """
    # Divides population into a number of 'shells'. Each shell contains the non-dominated solutions of Y.
    # Each recursive shell computes the non-dominated solutions of Y, excluding solutions already in shells.

    [n, m] = Y.shape
    S = np.zeros((n, 1))
    shell = np.zeros(n)
    dom_indices = []
    sub_indices = []

    for i in range(n):  # get number of points that dominate within Y
        current = Y[i, :]
        dom = False  # Does something dominate solution
        for j in range(n):
            dom = domination(Y[j, :], current)
            if dom:
                break
        if dom:
            S[i] = 0
            sub_indices.append(int(i))
        else:
            S[i] = 1
            dom_indices.append(int(i))
    shell[dom_indices] = index

    if sum(S) == n:  # if at last shell, terminate and chain back
        return shell
    else:
        shell[sub_indices] = recursive_pareto_shell_with_duplicates(Y[sub_indices, :], index+1)
        return shell


def domination(a, b):
    """
    Check if solution a dominates solution b (for minimising objective values)
    :param a:
    :param b:
    :return:
    """
    if np.array_equal(a, b):
        return False  # If a and b are equal, they do not dominate
    for i in range(len(a)):
        if a[i] > b[i]:
            return False  # At least one element in b is smaller than a
    # a and b cannot be equal here, and no element in b can be greater than a,
    # therefore at least one element in a must be bigger than b
    return True
