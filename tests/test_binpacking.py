from typing import Iterable
import pulp as pl
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from pulp2mat import get_vars, get_bounds, get_objective_array, get_constraint_matrix


class BinPackingProblem:
    def __init__(self, item_sizes: Iterable[int], bin_size):
        self.item_sizes = item_sizes
        self.num_items = len(self.item_sizes)
        self.num_bins = len(self.item_sizes)
        self.bin_size = bin_size
        self.problem = pl.LpProblem()
        self.__init_vars()
        self.__set_constraints()
        self.__set_objective()

    def __init_vars(self):
        # Variables
        self.x = {
            (i, j): pl.LpVariable("x_{}_{}".format(i, j), cat=pl.LpBinary)
            for i in range(self.num_items)
            for j in range(self.num_bins)
        }
        self.y = {
            j: pl.LpVariable("y_{}".format(j), cat=pl.LpBinary)
            for j in range(self.num_bins)
        }

    def __set_constraints(self):
        # Bin size constraints
        for j in range(self.num_bins):
            self.problem += (
                pl.lpSum(
                    self.x[i, j] * self.item_sizes[i] for i in range(self.num_items)
                )
                <= self.bin_size * self.y[j]
            )
        # One-hot constraint for each item
        for i in range(self.num_items):
            self.problem += pl.lpSum(self.x[i, j] for j in range(self.num_bins)) == 1

    def __set_objective(self):
        # Objective: minimize number of bins used.
        self.problem += pl.lpSum(self.y[j] for j in range(self.num_bins))

    def solve(self, solver: pl.LpSolver = None) -> pl.LpStatus:
        if solver is not None:
            stat = self.problem.solve(solver)
        else:
            stat = self.problem.solve()
        return stat


def test_binpack():
    """binpacking problem with number of item 10, bin size 10"""
    item_sizes = np.array([7, 3, 3, 1, 6, 8, 4, 9, 5, 2])
    bin_size = 10

    bpp = BinPackingProblem(item_sizes, bin_size)
    result = bpp.solve()

    assert pl.LpStatus[result] == "Optimal"
    assert bpp.problem.objective.value() == 5.0


def test_binpack2mat():
    """convert pulp binpacking problem into matrix formulation"""
    item_sizes = np.array([7, 3, 3, 1, 6, 8, 4, 9, 5, 2])
    bin_size = 10

    bpp = BinPackingProblem(item_sizes, bin_size)
    bpp.solve()
    # convert to matrix formulation
    obj_val_pulp = bpp.problem.objective.value()
    vars_dict, varnames = get_vars([bpp.x, bpp.y])
    const_mat, const_lb, const_ub = get_constraint_matrix(bpp.problem, vars_dict)
    obj_arr = get_objective_array(bpp.problem, vars_dict)
    integrality, lbounds, ubounds = get_bounds(vars_dict)
    # call scipy.optimize.milp
    bounds = Bounds(lbounds, ubounds)
    consts = LinearConstraint(const_mat, const_lb, const_ub)
    result = milp(c=obj_arr, constraints=consts, integrality=integrality, bounds=bounds)
    assert result.status == 0
    assert result.fun == obj_val_pulp
