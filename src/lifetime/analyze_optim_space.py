import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import DBSCAN
import functions as fc
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig
import hydra
import os

# Choose the experiment mode: "iso" (or "tunnel") vs "fading"
EXP_TYPE = "tl_fsm-13"  # "iso", "tl_clbr", "tl_fsm-13", "fading"
RUN_TYPE = "Opt"
# Bounds for the "fading" mode: [rho_prime, E_cb, E_loc, s, b, alpha, holes]
BOUNDS_FADING = [
    (1e-8, 1e-3),  # rho_prime
    (0.8,   2.2),  # E_cb
    (0.5,   1.8),  # E_loc
    (1e12, 1e14),  # s
    (1e10, 1e13),  # b
    (1e9,  5e10),  # alpha
    (1e2,    750)  # holes
]

# Select bounds based on experiment type
BOUNDS = BOUNDS_FADING
#BOUNDS = [(1e-8, 1e-3),(0.5,   1.8)]
# Objective function
def objective(params, cfg):
    mse_values = sim_starter(cfg, params, EXP_TYPE)
    return mse_values

def sim_starter(cfg, params, exp_type, PLOT:bool = False):
    # if exp_type == "iso":
    #     rho_prime = params[0]
    #     cfg.exp_type_fp.rho_prime = float(rho_prime)
    #     cfg.physics_fp.E_cb = float(params[1])
    #     MSE = fc.sim_lab_TL_residuals_iso(cfg,PLOT=PLOT)
    #     return MSE
    rho_prime, E_cb, E_loc, s, b, alpha, holes = params
    # Update the configuration with the new parameters
    cfg.exp_type_fp.rho_prime = float(rho_prime)
    cfg.physics_fp.E_cb = float(E_cb)
    cfg.physics_fp.E_loc = float(E_loc)
    cfg.physics_fp.s = float(s)
    cfg.physics_fp.b = float(b)
    cfg.physics_fp.alpha = float(alpha)
    cfg.exp_type_fp.holes = float(holes)     
    if exp_type == "iso":
        MSE = fc.sim_lab_TL_residuals_iso(cfg,PLOT=PLOT)
    elif exp_type == "tl_clbr":
        MSE = fc.sim_lab_TL_residuals(cfg, lab_data="CLBR_IRSL50_0.25KperGy",PLOT=PLOT)
    elif exp_type == "tl_fsm-13":
        MSE = fc.sim_lab_TL_residuals(cfg, lab_data="FSM-13_IRSL50_0.25KperGy",PLOT=PLOT)
    else:
        MSE = fc.sim_lab_TL_residuals_fading(cfg)  
    return MSE
 
# … all your existing imports and definitions …

def run_de(cfg):
    result = differential_evolution(
        objective,
        BOUNDS,
        args=(cfg,),
        strategy='best1bin',
        popsize=1,
        maxiter=1,
        polish=False,        # turn off the final “polish” so we see the raw population
        init='sobol',
        disp=False,
        workers=3
    )
    # result.population is an array of shape (NP, D)
    pop = result.population
    return result.x, result.fun, pop

def run_multistart_minimize(cfg, n_starts=10):
    best = []
    for i in range(n_starts):
        # random seed start inside your bounds
        x0 = np.array([np.random.uniform(lb, ub) for lb, ub in BOUNDS])
        res = minimize(
            lambda p: objective(p, cfg),
            x0,
            method='L-BFGS-B',
            bounds=BOUNDS,    # or 'L-BFGS-B' if you need bounds
            options={'gtol':1e-6, 'maxiter':10}
        )
        best.append((res.x, res.fun))
    # sort by objective value
    best.sort(key=lambda t: t[1])
    return best

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig):
    # 1) Run DE
    x_de, f_de, pop = run_de(cfg)
    print("DE best:", f_de, x_de)

    # 2) Cluster the DE population
    clustering = DBSCAN(eps=0.05, min_samples=2).fit(pop)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Found", n_clusters, "clusters in the final DE population")
    for lab in sorted(set(labels)):
        size = np.sum(labels == lab)
        print(f"  cluster {lab:2d}: {size} members")

    # 3) Multi‐start gradient runs
    grads = run_multistart_minimize(cfg, n_starts=30)
    print("\nTop 3 minima from random‐start gradient runs:")
    for xg, fg in grads[:3]:
        print("   ", fg, xg)

    # … then save CSVs or plots however you like …
    df_de = pd.DataFrame(pop, columns=[f"p{i}" for i in range(pop.shape[1])])
    df_de['cluster'] = labels
    df_de.to_csv("de_population_with_clusters.csv", index=False)
    print("Saved DE population + clustering to de_population_with_clusters.csv")

    return

if __name__ == "__main__":
    main()