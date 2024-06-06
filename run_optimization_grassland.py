from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
import numpy as np
import sys
from numpy import inf

import shutil
import pandas as pd

from pathlib import Path

from grassdt import pfts
from grassdt.io import read_grass_file, read_observation_file
from grassdt.run import run_grassmind
from grassdt.parameters import Parameters

def find_nearest_indices(a, v):
    a = a.to_numpy()
    v = v.to_numpy()
    idxs = np.searchsorted(a, v)
    next_idxs = idxs
    prev_idxs = idxs - 1

    flt = v - a[prev_idxs] < a[next_idxs] - v
    idxs[flt] = prev_idxs[flt]
    return idxs

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=16,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=np.array([0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2, 0.2, 100, 100, 100, 100]),
                         xu=np.array([2000, 2000, 2000, 2000, 1, 1, 1, 1, 2, 2, 2, 2, 15000, 15000, 15000, 15000]))

    def _evaluate(self, X, out, *args, **kwargs):
        pass  # Evaluation is done externally


algorithm = NSGA2(pop_size=300)

termination = get_termination("n_gen", 1000)


def external_evaluate(x, template_fpath, input_data_dpath, observation_fpath_template, workdir, keep_input_files, call_idx, key, plot, df_obs, available_pfts):

    x = x.reshape(4, 4, -1)
    n_samples = x.shape[-1]

    random_seeds = [1]

    p = Parameters.from_file(template_fpath)

    p['par.randomGeneratorSeedDisabled'] = 0
    p['par.maturityAges'] = [0.3, 0.3, 0.3, 0.8]
    p['par.seedPoolMortalityRates'] = [0.05, 0.5, 0.7, 0.8]
    p['par.seedGerminationRates'] = [1, 1, 1, 1]
    p['resultFileSwitch.grassplot'] = 0
    p['par.climateFilePath'] = [f'GCEF_Weather_EM_amb_2013_2022_{plot}.txt']

    # Create input files
    input_dpath = workdir / 'input'
    input_dpath.mkdir(parents=True, exist_ok=True)
    shutil.copytree(input_data_dpath, input_dpath, dirs_exist_ok=True)
    input_fpaths = []
    for s in range(n_samples):
        # Parameters to be optimized
        p['par.externalSeedInfluxPerHectare'] = x[0, :, s]
        p['par.backgroundMortalityParams'] = x[1, :, s]
        p['par.heightFromDbhParams'] = [x[2, :, s], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        p['par.laiFromDbhParams'] = [x[3, :, s], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        for r in random_seeds:
            p['randomGeneratorSeed'] = r
            fpath = input_dpath / (template_fpath.stem + f'_c{call_idx:06d}_s{s:06d}_r{r:06d}.par')
            p.write(fpath)
            input_fpaths.append(fpath)

        # Run in parallel
    output_dpath = workdir / 'results'
    output_dpath.mkdir(parents=True, exist_ok=True)
    run_grassmind(input_fpaths, log_fpath=workdir / 'parallel.log')

    # Remove input files
    if not keep_input_files:
        for fpath in input_fpaths:
            fpath.unlink()

    # Evaluate error
    mse_s = np.zeros(n_samples)
    for s in range(n_samples):

        # Read output files
        df_r = []
        for r in random_seeds:
            fpath = output_dpath / (template_fpath.stem + f'_c{call_idx:06d}_s{s:06d}_r{r:06d}.grassBioDT_calib')
            df = read_grass_file(fpath)
            fpath.unlink()
            df_r.append(df)
        df_all = pd.concat(df_r, keys=random_seeds, names=['r', 'id'])
        df_ave = df_all.groupby('id').mean()

        # Evaluate error

        # Find indices
        idxs = find_nearest_indices(df_ave['Time'], df_obs['Time'])
        for pft in available_pfts:
            if pft is None:
                full_key = key
            else:
                full_key = f'{key}.{pft}'

            vals = df_ave[full_key][idxs].to_numpy()
            refs = df_obs[full_key].to_numpy()

            flt = np.isnan(vals)
            vals[flt] = np.inf

            mse_s[s] += np.sum((vals - refs)**2)

    if n_samples == 1:
        return mse_s[0]
    return mse_s.reshape(-1, 1)
        

########################################################

def main(template_fpath, input_data_dpath, observation_fpath_template, workdir, keep_input_files):

    problem = MyProblem()

    # create an algorithm object that never terminates
    algorithm.setup(problem, termination=termination)

    key = 'Biomass'
    plot = '011'

    # Read observation data
    df_obs = read_observation_file(observation_fpath_template, key=key, plot=plot)
    if key == 'Biomass':
        available_pfts = pfts
    elif key == 'Cover':
        available_pfts = pfts[:-1]
    else:
        available_pfts = [None]

    call_idx = -1
  
    # Variables to keep track of the best solution found
    global_best_X = None
    global_best_F = np.inf

    for n_gen in range(50):
    
        # Step 1: Generate a new population
        pop = algorithm.ask()

        call_idx += 1
        F = external_evaluate(pop.get("X"), template_fpath, input_data_dpath, observation_fpath_template, workdir, keep_input_files, call_idx, key, plot, df_obs, available_pfts)

        # replace infinity values
        F[F == inf] = sys.float_info.max/2
        F[F == -inf] = -sys.float_info.max/2

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('template_fpath', type=Path)
    parser.add_argument('input_data_dpath', type=Path)
    parser.add_argument('observation_fpath_template')
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--keep_input_files', action='store_true')
    kwargs = vars(parser.parse_args())
    main(**kwargs)



