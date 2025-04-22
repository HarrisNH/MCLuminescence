import numpy as np
import functions as fc
import pandas as pd
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
import hydra
from scipy.optimize import differential_evolution
import os

# Choose the experiment mode: "iso" (or "tunnel") vs "fading"
EXP_TYPE = "tl_clbr"  # "iso", "tl_clbr", "tl_fsm-13", "fading"
RUN_TYPE = "Train"
RUNS = 5
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
 

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    if RUN_TYPE == "Train":
        for i in range(RUNS):
            # Create a partial function where cfg is fixed.  
            result = differential_evolution(objective, BOUNDS, args=(cfg,), 
                                            strategy='randtobest1bin', 
                                            init="sobol", mutation=0.5, recombination=0.3,
                                             maxiter = 5, popsize=10, 
                                            workers=1, tol=1e-7, disp=True, polish=True)
            
            # Print the result
            print("Optimal parameters found:", result.x)
            print("Optimal MSE:", result.fun)

            # build a one‑row DataFrame: all params + final MSE
            columns = [f"param_{i}" for i in range(len(result.x))] + ["mse"]
            data    = [list(result.x) + [float(result.fun)]]
            df      = pd.DataFrame(data, columns=columns)
            out_path = os.path.join(PROJECT_ROOT, f"results/sims/result_{EXP_TYPE}.csv")
            prev_results = pd.read_csv(out_path)
            all_results = pd.concat([df,prev_results])
            all_results.to_csv(out_path, index=False)
            print(f"Saved CSV to {out_path}, now creating plot with best parameters")
            # Plot the simulation with the best parameters
            if EXP_TYPE != "iso":
                sim_starter(cfg, result.x, exp_type=EXP_TYPE, PLOT=True)
            else:
                sim_starter(cfg, result.x, exp_type=EXP_TYPE, PLOT=True)
    elif RUN_TYPE == "Opt":
        print("Just using optimal parameters from previously saved run")
        result = pd.read_csv(os.path.join(PROJECT_ROOT, f"results/sims/result_{EXP_TYPE}.csv"))
        result_best = result[result["mse"]==result.mse.min()]
        result_mean = np.mean(result,axis=0)
        new_params = np.array(result_best.iloc[0].iloc[0:-1].values)
        sim_starter(cfg, new_params, exp_type=EXP_TYPE, PLOT=True)
    else:
        print("Just using optimal parameters from previously saved run")
        result = pd.read_csv(os.path.join(PROJECT_ROOT, f"results/sims/result_{EXP_TYPE}.csv"))
        for i in range(1, len(result.columns)-1):
            new_params = np.array(result.iloc[i].iloc[0:-1].values)
            sim_starter(cfg, new_params, exp_type=EXP_TYPE, PLOT=True)
    return result

if __name__ == "__main__":
    res = main()
    print(res)

    ##Optimal parameters found: [3.86414531e-03 2.45717498e+00 5.44385880e+22 1.67193893e+10 4.90544184e+02]



