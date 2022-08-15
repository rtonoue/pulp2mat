# pulp2mat

Convert pulp model into matrix formulation.

It can be easily thrown to scipy.optimize.milp function.


# How to install

- Clone this repository.

```
$ git clone https://github.com/rtonoue/pulp2mat.git
$ cd pulp2mat
```

- install dependencies

poetry users can install all dependencies easily.

```
$ poetry install
```

Without poetry, please look at pyproject.toml and install all dependencies manually. 

# Quick Example

All variables must be defined in dictionaries. The key is tuple of variable indices, the value is pulp.LpVariable.

For example, the binpacking problem can be formulated with pulp as below;

```python
import pulp as pl
import numpy as np

item_sizes = np.array([7, 3, 3, 1, 6, 8, 4, 9, 5, 2])
num_items = len(item_sizes)
num_bins = len(item_sizes)
bin_size = 10

# Variables * must be defined as dictionaries
x = {
    (i, j): pl.LpVariable("x_{}_{}".format(i, j), cat=pl.LpBinary)
    for i in range(num_items)
    for j in range(num_bins)
}
y = {
    j: pl.LpVariable("y_{}".format(j), cat=pl.LpBinary)
    for j in range(num_bins)
}

problem = pl.LpProblem()

# Bin size constraint for each bin
for j in range(num_bins):
    problem += (
        pl.lpSum(
            x[i, j] * item_sizes[i] for i in range(num_items)
        )
        <= bin_size * y[j]
    )
# One-hot constraint for each item
for i in range(num_items):
    problem += pl.lpSum(x[i, j] for j in range(num_bins)) == 1

# Objective: minimize number of bins used.
problem += pl.lpSum(y[j] for j in range(num_bins))
```

the ```pulp.LpProblem``` object and the list of variable dictionaries can be converted to the matrix format for ```scipy.optimize.milp```.

```python
import pulp2mat
from scipy.optimize import milp
c, integrality, constraints, bounds = pulp2mat.convert_all(problem, [x, y])
result = milp(c, integrality=integrality, constraints=constraints, bounds=bounds)
```
