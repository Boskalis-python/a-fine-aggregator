"""
Script containing the a-fine aggregation algorithm.

Copyright (c) 2024 Royal Boskalis
"""
from typing import Union

import numpy as np


def a_fine_aggregator(
    w: Union[list[float], np.ndarray[float]],
    p: Union[list[list[Union[float, int]]], np.ndarray[list[Union[float, int]]]],
) -> np.ndarray[float]:
    """
    Function for aggregating scores in affine spaces, by means of the least square
    distance minimization.

    :param w: weights of the different objectives
    :param p: 2d-array with the scores of the objectives. n-by-m, where n is the number
        of objectives and m the population size
    :return: ndarray with the aggregated scores
    """
    assert len(w) == len(p), (
        f"The number of weights ({len(w)}) is not equal to the number of objectives "
        f"({len(p)})."
    )
    assert (
        round(sum(w), 4) == 1
    ), f"The sum of the weights ({round(sum(w), 4)}) is not equal to 1."

    # catch the edge cases and return an array filled with -50.0
    m_unique = len(np.unique(np.array(p), axis=1)[0])
    n_unique = len(np.unique(np.array(p), axis=1))
    if m_unique == 1 or m_unique == n_unique == 2:
        return np.full(len(p[0]), -50.0, dtype=float)

    # transpose the array to make further calculations easier
    p_transposed = np.array(p).transpose()

    # calculate the standard deviation per criteria. If std == 0, a value << 1 is
    # inserted to prevent divide by zero error
    std = np.std(p_transposed, axis=0)
    std[std == 0] = 1e-6

    # calculate the z-score normalized scores per criteria
    z = (p_transposed - np.mean(p_transposed, axis=0)) / std

    # calculate representative preference scores (P_i^*)
    p_star = np.sum(w * z, axis=1)

    # return min-max normalized results, so everything is on the scale [0-100]
    return -1 * (p_star - min(p_star)) / (max(p_star) - min(p_star)) * 100
