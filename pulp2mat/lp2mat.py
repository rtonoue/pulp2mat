from typing import Iterable, Tuple

import pulp as pl
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult


def get_vars(
    all_vars: Iterable[dict[Tuple, pl.LpVariable]]
) -> Tuple[dict[str, Tuple[int, int, np.float64, np.float64]], list[str]]:
    vars_dict = {}
    varnames = []
    idx = 0
    for variables in all_vars:
        for key, itm in variables.items():
            if itm.cat == "Continuous":
                cat = 0
            elif itm.cat == "Integer":
                cat = 1
            else:
                raise ValueError(
                    "LpVariable category needs to be Continuous or Integer."
                )
            lb = itm.lowBound
            ub = itm.upBound
            if lb is None or np.isnan(lb):
                lb = -np.inf
            if ub is None or np.isnan(ub):
                ub = np.inf
            vars_dict[itm.name] = (idx, cat, lb, ub)
            varnames.append(itm.name)
            idx += 1
    return (vars_dict, varnames)


def get_constraint_matrix(
    problem: pl.LpProblem, vars_dict: dict[str, Tuple[int, int, np.float64, np.float64]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get constraint matrix c and bound arrays b_l, b_u such that;
    b_l <= c.dot(x) <= b_u

    Args:
        problem (pl.LpProblem): pulp model to be converted
        vars_dict (dict[str, Tuple[int, int, np.float64, np.float64]]): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: constraint matrix c, bound arrays b_l, b_u
    """
    n_col = len(vars_dict)
    n_row = len(problem.constraints)
    const_mat = np.zeros((n_row, n_col), dtype=np.float64)
    const_lb = np.zeros(n_row, dtype=np.float64)
    const_ub = np.zeros(n_row, dtype=np.float64)
    for i, key in enumerate(problem.constraints.keys()):
        const_coefs = problem.constraints[key].to_dict()
        const_ub[i] = problem.constraints[key].getUb()
        const_lb[i] = problem.constraints[key].getLb()
        if np.isnan(const_ub[i]):
            const_ub[i] = np.inf
        if np.isnan(const_lb[i]):
            const_lb[i] = -np.inf
        for coef in const_coefs:
            name = coef["name"]
            val = coef["value"]
            const_mat[i, vars_dict[name][0]] = val
    return (const_mat, const_lb, const_ub)


def get_objective_array(
    problem: pl.LpProblem, vars_dict: dict[str, Tuple[int, int, np.float64, np.float64]]
) -> np.ndarray:
    n_col = len(vars_dict)
    obj_arr = np.zeros(n_col, dtype=np.float64)
    obj_coefs = problem.objective.to_dict()
    for coef in obj_coefs:
        name = coef["name"]
        val = coef["value"]
        obj_arr[vars_dict[name][0]] = val
    return obj_arr


def get_bounds(
    vars_dict: dict[str, Tuple[int, int, np.float64, np.float64]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get 1-d arrays of l and u such that
    l <= x <= u

    Args:
        vars_dict (dict[str, Tuple[int, int, np.float64, np.float64]]):
            key; variable name, value; tuple of variable ID, integrality, lowbound, upbound.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: integrality, lowbound, upbound
    """
    n_col = len(vars_dict)
    lbounds = np.zeros(n_col, dtype=np.float64)
    ubounds = np.zeros(n_col, dtype=np.float64)
    integrality = np.zeros(n_col, dtype=np.float64)
    for key, itm in vars_dict.items():
        lbounds[itm[0]] = itm[2]
        ubounds[itm[0]] = itm[3]
        integrality[itm[0]] = itm[1]
    return (integrality, lbounds, ubounds)


def convert_all(
    problem: pl.LpProblem, all_vars: Iterable[dict[Tuple, pl.LpVariable]]
) -> Tuple[np.ndarray, np.ndarray, LinearConstraint, Bounds]:
    vars_dict, varnames = get_vars(all_vars)
    const_mat, const_lb, const_ub = get_constraint_matrix(problem, vars_dict)
    obj_arr = get_objective_array(problem, vars_dict)
    integrality, lbounds, ubounds = get_bounds(vars_dict)
    # call scipy.optimize.milp
    bounds = Bounds(lbounds, ubounds)
    consts = LinearConstraint(const_mat, const_lb, const_ub)
    return (obj_arr, integrality, consts, bounds)


def decode_solution(
    res: OptimizeResult,
    problem: pl.LpProblem,
    all_vars: Iterable[dict[Tuple, pl.LpVariable]],
):
    _, varnames = get_vars(all_vars)
    assert len(res.x) == len(varnames)
    values = dict()
    for value, name in zip(res.x, varnames):
        values[name] = value
    problem.assignVarsVals(values)
