import numpy as np
import matplotlib as mpl
import pandas as pd
import shutil
import sys

from matplotlib import pyplot as plt
from pathlib import Path


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.problem import Problem


def find_nearest_indices(a, v):
    a = a.to_numpy()
    v = v.to_numpy()
    idxs = np.searchsorted(a, v)
    next_idxs = idxs
    prev_idxs = idxs - 1

    flt = v - a[prev_idxs] < a[next_idxs] - v
    idxs[flt] = prev_idxs[flt]
    return idxs



def evaluate_grassmind(x, template_fpath, input_data_dpath, observation_fpath_template, workdir, keep_input_files, key, plot, df_obs, available_pfts, call_idx):
    from grassdt.io import read_grass_file
    from grassdt.run import run_grassmind
    from grassdt.parameters import Parameters

    if x.ndim == 1:
        x = x[np.newaxis]

    n_samples = x.shape[0]

    random_seeds = [1]
    #random_seeds = list(range(1, 161))

    p = Parameters.from_file(template_fpath)

    p['par.randomGeneratorSeedDisabled'] = 0
    p['par.maturityAges'] = [0.3, 0.3, 0.3, 0.8]
    p['par.seedPoolMortalityRates'] = [0.05, 0.5, 0.7, 0.8]
    p['par.seedGerminationRates'] = [1, 1, 1, 1]
    p['resultFileSwitch.grassplot'] = 0
    p['par.climateFilePath'] = [f'GCEF_Weather_EM_amb_2013_2022_{plot}.txt']

    # Default values
    p['par.externalSeedInfluxPerHectare'] = [500, 500, 500, 500]
    p['par.backgroundMortalityParams'] = [0.2, 0.3, 0.9, 0.07]
    p['par.heightFromDbhParams'] = [[0.6, 1.7, 0.7, 1.2],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 0.0]]
    p['par.laiFromDbhParams'] = [[8500, 1800, 3900, 1900],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]]

    # Create input files
    input_dpath = workdir / 'input'
    input_dpath.mkdir(parents=True, exist_ok=True)
    shutil.copytree(input_data_dpath, input_dpath, dirs_exist_ok=True)
    input_fpaths = []
    for s in range(n_samples):
        # Parameters to be optimized
        # Test: only 2D case
        par = [500, 500, 500, 500]
        par[0] = x[s, 0]
        par[1] = x[s, 1]
        p['par.externalSeedInfluxPerHectare'] = par
        # Optimize all (16D case)
        # p['par.externalSeedInfluxPerHectare'] = x[0, :, s]
        # p['par.backgroundMortalityParams'] = x[1, :, s]
        # p['par.heightFromDbhParams'] = [x[2, :, s], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        # p['par.laiFromDbhParams'] = [x[3, :, s], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
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



class MyProblem(Problem):
    def __init__(self, grassland_dpath):
        from grassdt import pfts
        from grassdt.io import read_observation_file

        super().__init__(
            n_obj=1,
            n_ieq_constr=0,
            # 16D case
            # n_var=16,
            # xl=np.array([   0,    0,    0,    0, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2, 0.2,   100,   100,   100,   100]),
            # xu=np.array([2000, 2000, 2000, 2000,    1,    1,    1,    1,   2,   2,   2,   2, 15000, 15000, 15000, 15000]),
            # 2D case test
            n_var=2,
            xl=np.array([   0,    0]),
            xu=np.array([1000, 1000]),
        )

        self.call_idx = 0


        self.template_fpath = grassland_dpath / 'par_file_templates/GCEF_EM_amb_2013_2022.tag'
        self.input_data_dpath = grassland_dpath / 'task_fileset'
        self.observation_fpath_template = str(grassland_dpath / 'observation_data/GCEF_{key}_EM_amb_2013_2022_{plot}.txt')
        self.workdir = Path('work')
        self.keep_input_files = True

        self.key = 'Biomass'
        self.plot = 'mean'

        # Read observation data
        self.df_obs = read_observation_file(self.observation_fpath_template, key=self.key, plot=self.plot)
        if self.key == 'Biomass':
            self.available_pfts = pfts
        elif self.key == 'Cover':
            self.available_pfts = pfts[:-1]
        else:
            self.available_pfts = [None]

    def evaluate_error(self, X, *args, **kwargs):
        return evaluate_grassmind(X, self.template_fpath, self.input_data_dpath, self.observation_fpath_template, self.workdir, self.keep_input_files, self.key, self.plot, self.df_obs, self.available_pfts, call_idx=self.call_idx)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.evaluate_error(X, *args, **kwargs)
        self.call_idx += 1


def main(grassland_dpath):
    sys.path.append(str(grassland_dpath / 'src'))  # XXX FIXME
    problem = MyProblem(grassland_dpath)

    algorithm = NSGA2(pop_size=10)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=False)

    print("Optimal point:", res.X)
    print("value:        ", res.F)
    print("value check:  ", problem.evaluate_error(res.X))

    # Evaluate on grid for plotting
    fpath = Path("gl.npz")
    if fpath.exists():
        npz = np.load(fpath)
        x = npz['x']
        y = npz['y']
        F = npz['F']
    else:
        xl, xu = 0, 1000
        yl, yu = 0, 1000
        x, y = np.meshgrid(np.linspace(xl, xu, 20), np.linspace(yl, yu, 20))
        X = np.asarray([np.ravel(x), np.ravel(y)]).T
        F = problem.evaluate_error(X).reshape(x.shape)
        print(X.shape, F.shape)
        np.savez(fpath, x=x, y=y, F=F)

    # Plot
    plt.figure()
    plt.pcolormesh(x, y, F, vmin=1e5, vmax=3e5)
    plt.scatter(*res.X)
    plt.colorbar()
    plt.savefig('gl.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('grassland_dpath', type=Path)
    kwargs = vars(parser.parse_args())
    main(**kwargs)
