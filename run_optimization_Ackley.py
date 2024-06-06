from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
import numpy as np
import sys
from numpy import inf

import shutil
import pandas as pd

from pathlib import Path


# Define the Ackley function for external evaluation
def ackley_function(X):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = X.shape[1]

    sum1 = np.sum(X**2, axis=1)
    sum2 = np.sum(np.cos(c * X), axis=1)
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    
    F = term1 + term2 + a + np.exp(1)
    
    # Ensure F is a 2D array
    return F[:, np.newaxis]


# Define a custom problem class that pymoo will use
class MyAckleyProblem(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-32.768, xu=32.768)

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluation is done externally, so this can be left as pass or a placeholder
        pass

algorithm = NSGA2(pop_size=300)

termination = get_termination("n_gen", 1000)


def main():

    problem = MyAckleyProblem()

    # create an algorithm object that never terminates
    algorithm.setup(problem, termination=termination)
  
    # Variables to keep track of the best solution found
    global_best_X = None
    global_best_F = np.inf

    for n_gen in range(50):
    
        # Step 1: Generate a new population
        pop = algorithm.ask()

        F = ackley_function(pop.get("X"))
  
        # Step 3: Set the evaluated results back to the population
        pop.set("F", F)

        # Step 4: Provide the evaluated results back to the algorithm
        algorithm.tell(infills=pop)

        # Update the global best solution
        best_idx = np.argmin(F)
        if F[best_idx] < global_best_F:
            global_best_F = F[best_idx]
            global_best_X = pop.get("X")[best_idx]

        # Print the ongoing best values found so far
        print(f"Generation {algorithm.n_gen-1}: Best F = {global_best_F}, Best X = {global_best_X}")
        print("")


    # Get the final results
    res = algorithm.result()

    # The final population and their evaluated objectives
#    print("Resulted values:", res.X)

########################################################

if __name__ == '__main__':
    main()



