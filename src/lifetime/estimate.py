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

bounds = [(1e-7, 1e-3),(0.1,3),(1e12,1e16),(1e9,3e10),(200,100)]

# Objective function
def objective(params,cfg):
    mse_values = [sim_starter(cfg,params)]
    return mse_values

def sim_starter(cfg,params):
    # Extract the parameters from the input list
    rho_prime, E, b, alpha,holes = params

    # Update the configuration with the new parameters
    cfg.exp_type_fp.rho_prime = float(rho_prime)
    cfg.physics_fp.E = float(E)
    cfg.physics_fp.b = float(b)
    cfg.physics_fp.alpha = float(alpha)
    cfg.exp_type_fp.holes = float(holes)
    # Print the updated configuration for debugging

    # Run the simulation with the updated parameters
    MSE = fc.sim_lab_TL_residuals_iso(cfg)

    return MSE



@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    # Create a partial function where cfg is fixed.    
    result = differential_evolution(objective, bounds, args =(cfg,), strategy='best1bin',init = "sobol", maxiter=5, popsize=5, tol=1e-4, workers = 1, disp=True,polish=True)

    print("Optimal parameters found:", result.x)
    print("Optimal MSE:", result.fun)
    return result
if __name__ == "__main__":
    res = main()
    print(res)