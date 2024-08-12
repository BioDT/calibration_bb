import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.heatmap import Heatmap


# Fix matplolib compatibility issue in pymoo
mpl.cm.get_cmap = plt.get_cmap

# Define the Ackley function
def ackley_function(X):
    if X.ndim == 1:
        X = X[np.newaxis]

    a = 20
    b = 0.2
    c = 2 * np.pi
    d = X.shape[1]

    sum1 = np.sum(X**2, axis=1)
    sum2 = np.sum(np.cos(c * X), axis=1)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    F = term1 + term2 + a + np.exp(1)
    return F


# Define a problem class that pymoo will use
class MyAckleyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=-32.768, xu=32.768)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = ackley_function(x)


def main():
    problem = MyAckleyProblem()
    algorithm = NSGA2(pop_size=300)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=1,
                   verbose=False)

    print("Optimal point:", res.X)
    print("value:        ", res.F)
    print("value check:  ", ackley_function(res.X))

    # Evaluate on grid for plotting
    xl, xu = -5, 5
    yl, yu = -5, 5
    x, y = np.meshgrid(np.linspace(xl, xu, 100), np.linspace(yl, yu, 100))
    X = np.asarray([np.ravel(x), np.ravel(y)]).T
    F = ackley_function(X).reshape(x.shape)
    print(X.shape, F.shape)

    # Plot
    plt.figure()
    plt.pcolormesh(x, y, F)
    plt.scatter(*res.X)
    plt.colorbar()
    plt.savefig('test.png')

    # Pymoo functions
    # plot = Heatmap()
    # plot.add(F)
    # plot.save("heatmap.png")

    # plot = Scatter()
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    # plot.add(res.F, facecolor="none", edgecolor="red")
    # plot.save("scatter.png")


if __name__ == '__main__':
    main()
