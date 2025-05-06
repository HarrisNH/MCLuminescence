import numpy as np
import functions as fc
import pandas as pd
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
import hydra
from scipy.optimize import differential_evolution
import os

# Choose the experiment mode and run type:
EXP_TYPE = "tl_clbr"  # "iso", "tl_clbr", "tl_fsm-13"
RUN_TYPE = "Tra3in"
RUNS = 15

# Bounds for the "fading" mode: [rho_prime, E_cb, E_loc, s, b, alpha, holes]
BOUNDS= [
    (1e-8, 1e-3),  # rho_prime
    (1.9,   2.4),  # E_cb
    (1.2,   1.7),  # E_loc_1
    (1,   1.5),  # E_loc_2
    (100,   400), #D0
    (1e12, 1e14),  # s
    (1e10, 1e13),  # b
    (1e9,  5e10),  # alpha
    (1e2,    750),  # holes
    (0,1) # retrapping probability
]

# Objective function
def objective(params, cfg):
    mse_values = sim_starter(cfg, params, EXP_TYPE)
    return mse_values

def sim_starter(cfg, params, exp_type, PLOT:bool = False):
    rho_prime, E_cb, E_loc_1,E_loc_2, D0, s, b, alpha, holes,retrap = params
    # Update the configuration with the new parameters
    cfg.exp_type_fp.rho_prime = float(rho_prime)
    cfg.physics_fp.E_cb = float(E_cb)
    cfg.physics_fp.E_loc_1 = float(E_loc_1)
    cfg.physics_fp.E_loc_2 = float(E_loc_2)
    cfg.physics_fp.D0 = float(D0)
    cfg.physics_fp.s = float(s)
    cfg.physics_fp.b = float(b)
    cfg.physics_fp.alpha = float(alpha)
    cfg.exp_type_fp.holes = float(holes)    
    cfg.physics_fp.Retrap = float(retrap)

    if exp_type == "iso":
        MSE = fc.sim_lab_TL_residuals_iso(cfg,PLOT=PLOT)
    elif exp_type == "tl_clbr":
        MSE = fc.sim_lab_TL_residuals(cfg, lab_data="CLBR_IRSL50_0.25KperGy",PLOT=PLOT)
    elif exp_type == "tl_fsm-13":
        MSE = fc.sim_lab_TL_residuals(cfg, lab_data="FSM-13_IRSL50_0.25KperGy",PLOT=PLOT)
    else: 
        raise ValueError(f"Unknown experiment type: {exp_type}")
    return MSE
 

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    if RUN_TYPE == "Train":
        for i in range(RUNS):
            result = differential_evolution(objective, BOUNDS, args=(cfg,), 
                                            strategy='randtobest1bin', 
                                            init="sobol", mutation=0.5, recombination=0.3,
                                             maxiter = 7, popsize=15, 
                                            workers=5, tol=1e-7, disp=True, polish=True)
            
            # Print the result
            print("Optimal parameters found:", result.x)
            print("Optimal MSE:", result.fun)

            # build a one‑row DataFrame: all params + final MSE
            columns = [f"param_{i}" for i in range(len(result.x))] + ["mse"]
            data    = [list(result.x) + [float(result.fun)]]
            df      = pd.DataFrame(data, columns=columns)
            out_path = os.path.join(PROJECT_ROOT, f"results/lab_sims/result_{EXP_TYPE}.csv")
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
        result = pd.read_csv(os.path.join(PROJECT_ROOT, f"results/lab_sims/result_{EXP_TYPE}.csv"))
        result_best = result[result["mse"]==result.mse.min()]
        new_params = np.array(result_best.iloc[0].iloc[0:-1].values)
        sim_starter(cfg, new_params, exp_type=EXP_TYPE, PLOT=True)
    return result

if __name__ == "__main__":
    res = main()
    print(res)


