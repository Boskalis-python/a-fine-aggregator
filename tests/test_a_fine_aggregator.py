"""
Test scripts for the a-fine aggregator algorithm.

Copyright (c) 2024 Royal Boskalis
"""
from numpy.testing import assert_allclose

from algorithm import a_fine_aggregator


def test_to_known_outcome():
    """
    Test to see if the aggregation algorithm is broken. The comparison is made with a
    known result. Trows AssertionError when the result is not the same.

    :return: None
    """

    data = [
        [20, 95, 50, 65, 15],
        [40, 35, 60, 35, 25],
        [60, 65, 10, 65, 75],
        [80, 15, 15, 35, 95],
    ]

    desired_result = [-35.85212, -100.0000, -0.0000, -74.66776, -40.68803]

    w = [0.35, 0.15, 0.45, 0.05]
    ret = a_fine_aggregator(w=w, p=data)

    assert_allclose(
        actual=ret,
        desired=desired_result,
        rtol=1e-4,
        atol=5e-4,
        err_msg="Returned data is wrong",
    )
    return


def test_edge_cases():
    """
    Test to see if the aggregation algorithm will return the correct value for edge
    cases. Trows AssertionError when the results are different then expected.

    :return: None
    """

    three_equal = [
        [15, 15, 15],
        [95, 95, 95],
    ]
    only_one_member = [
        [15],
        [95],
    ]
    only_two_members = [
        [15, 65],
        [95, 45],
    ]
    only_two_members_three_crit = [
        [15, 65],
        [95, 45],
        [55, 85],
    ]

    three_equal_result = [-50.0000, -50.0000, -50.0000]
    only_one_member_result = [-50.0000]
    only_two_members_result = [-50.0000, -50.0000]
    only_two_members_three_crit_result = [-0.0000, -100.0000]

    w_1 = [0.50, 0.50]
    w_2 = [1 / 3, 1 / 3, 1 / 3]
    three_equal_ret = a_fine_aggregator(w=w_1, p=three_equal)
    only_one_member_ret = a_fine_aggregator(w=w_1, p=only_one_member)
    only_two_members_ret = a_fine_aggregator(w=w_1, p=only_two_members)
    only_two_members_three_crit_ret = a_fine_aggregator(
        w=w_2, p=only_two_members_three_crit
    )

    assert_allclose(
        actual=three_equal_ret,
        desired=three_equal_result,
        rtol=1e-4,
        atol=5e-4,
        err_msg="Returned data is wrong - three equal members",
    )
    assert_allclose(
        actual=only_one_member_ret,
        desired=only_one_member_result,
        rtol=1e-4,
        atol=5e-4,
        err_msg="Returned data is wrong - only one members",
    )
    assert_allclose(
        actual=only_two_members_ret,
        desired=only_two_members_result,
        rtol=1e-4,
        atol=5e-4,
        err_msg="Returned data is wrong - only two members",
    )
    assert_allclose(
        actual=only_two_members_three_crit_ret,
        desired=only_two_members_three_crit_result,
        rtol=1e-4,
        atol=5e-4,
        err_msg="Returned data is wrong - only two members three crit",
    )
    return


if __name__ == "__main__":
    test_to_known_outcome()
    test_edge_cases()
