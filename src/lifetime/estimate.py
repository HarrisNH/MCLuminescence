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

# Choose the experiment mode: "iso" (or "tunnel") vs "fading"
EXP_TYPE = "fading"  

# Bounds for the "iso" (or tunnel) mode: [rho_prime, E_cb/E_loc unused, b, alpha, holes]
BOUNDS = [
    (1e-7, 1e-2),   # rho_prime
    (0.1,    3.0),  # E (single trap depth)
    (1e11,  1e17),  # b (tunnelling prefactor)
    (1e9,   5e10),  # alpha (tunnelling inverse length)
    (100,    750)   # holes density parameter
]

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
BOUNDS = BOUNDS if EXP_TYPE in ["iso", "tunnel"] else BOUNDS_FADING

# Objective function
def objective(params, cfg):
    mse_values = [sim_starter(cfg, params, EXP_TYPE)]
    return mse_values

def sim_starter(cfg, params, exp_type):
    # Extract the parameters from the input list
    if EXP_TYPE in ["tunnel", "iso"]:
        rho_prime, E, b, alpha, holes = params
        # Update the configuration with the new parameters
        cfg.exp_type_fp.rho_prime = float(rho_prime)
        cfg.physics_fp.E = float(E)
        cfg.physics_fp.b = float(b)
        cfg.physics_fp.alpha = float(alpha)
        cfg.exp_type_fp.holes = float(holes)
        # Print the updated configuration for debugging
        if exp_type == "ISO":
            MSE = fc.sim_lab_TL_residuals_iso(cfg)
        # Run the simulation with the updated parameters
        else: MSE = fc.sim_lab_TL_residuals(cfg)

    # For the "fading" mode
    else:
        rho_prime, E_cb, E_loc, s, b, alpha, holes = params
        # Update the configuration with the new parameters
        cfg.exp_type_fp.rho_prime = float(rho_prime)
        cfg.physics_fp.E_cb = float(E_cb)
        cfg.physics_fp.E_loc = float(E_loc)
        cfg.physics_fp.s = float(s)
        cfg.physics_fp.b = float(b)
        cfg.physics_fp.alpha = float(alpha)
        cfg.exp_type_fp.holes = float(holes)     
        MSE = fc.sim_lab_TL_residuals_fading(cfg)
    return MSE

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:

    # Create a partial function where cfg is fixed.  
    result = differential_evolution(objective, BOUNDS, args=(cfg,), strategy='best1bin', init="sobol", maxiter=7, popsize=7, tol=1e-7, workers=1, disp=True, polish=True)
    
    # Print the result
    print("Optimal parameters found:", result.x)
    print("Optimal MSE:", result.fun)
    return result

if __name__ == "__main__":
    res = main()
    print(res)

    ##Optimal parameters found: [3.86414531e-03 2.45717498e+00 5.44385880e+22 1.67193893e+10 4.90544184e+02]