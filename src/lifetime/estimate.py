import numpy as np
import functions as fc
import pandas as pd
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import hydra
from functools import partial
from scipy.optimize import differential_evolution
import os

bounds = [(1e-7, 1e-2),(0.1,3),(1e12,1e24),(1e9,5e10),(200,1500)]
bounds_TL = [(1e-7, 1e-2),(0.1,3),(1e12,1e24),(1e9,5e10),(200,1500)]

# Objective function
def objective(params,cfg):
    mse_values = [sim_starter(cfg,params,"ISO")]
    return mse_values

# Objective function
def objective_TL(params,cfg):
    mse_values = [sim_starter(cfg,params,"TL")]
    return mse_values

def sim_starter(cfg,params,exp_type):
    # Extract the parameters from the input list
    rho_prime, E, b, alpha,holes = params

    # Update the configuration with the new parameters
    cfg.exp_type_fp.rho_prime = float(rho_prime)
    cfg.physics_fp.E = float(E)
    cfg.physics_fp.b = float(b)
    cfg.physics_fp.alpha = float(alpha)
    cfg.exp_type_fp.holes = float(holes)
    # Print the updated configuration for debugging
    if exp_type == "ISO":
        MSE = fc.sim_lab_TL_residuals(cfg)
    # Run the simulation with the updated parameters
    else: MSE = fc.sim_lab_TL_residuals_iso(cfg)

    return MSE



@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    # Create a partial function where cfg is fixed.    
    result_ISO = differential_evolution(objective, bounds, args =(cfg,), strategy='best1bin',init = "sobol", maxiter=7, popsize=7, tol=1e-7, workers = 1, disp=True,polish=True)
    result_TL = differential_evolution(objective_TL,bounds_TL, args =(cfg,), strategy='best1bin',init = "sobol", maxiter=2, popsize=2, tol=1e-7, workers = 1, disp=True,polish=True)
    print("Optimal parameters found:", result_TL.x)
    print("Optimal MSE:", result_TL.fun)
    return result_TL
if __name__ == "__main__":
    res = main()
    print(res)

    ##Optimal parameters found: [3.86414531e-03 2.45717498e+00 5.44385880e+22 1.67193893e+10 4.90544184e+02]