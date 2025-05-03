import pandas as pd
from paths import PROJECT_ROOT
import numpy as np

def extract_experiment_data(cfg):
    """
    Extract data from excel sheet with experimental data
    """
    # Read the CSV file
    data = pd.read_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IRSL50_0.25KperGy.csv")    
    return data[["T_end","Fill"]]

def save_sim_results(cfg,e_ratio):
    results = pd.DataFrame(columns=["T_end","Fill"])
    for i in range(e_ratio.shape[-1]):
        # Save the simulated data to a CSV file
        idx = np.argmin(e_ratio[:,:,i]!=0)
        T_end = cfg.exp_type_fp.T_end[i]
        ratio = e_ratio[idx-1,:,i][0] 
        results = pd.concat([results,pd.DataFrame([[T_end,ratio]], columns=results.columns)], ignore_index=True)

    results.to_csv(f"{PROJECT_ROOT}/results/sims/simulated_data.csv", index=False)
        
    

def main(cfg,x_ax, lum, e_ratio,configs):
    """
    Main function to analyze and compare expirimental and simulated results
    """
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    exp_data = extract_experiment_data(cfg)
    save_sim_results(cfg,e_ratio)
    sim_data = pd.read_csv(f"{PROJECT_ROOT}/results/sims/simulated_data.csv")
    diff = (exp_data["Fill"]-sim_data["Fill"])**2    
    MSE = np.mean(diff)
    errors = pd.DataFrame(columns=["rho_prime","E","s","N_e","MSE"])
    errors = pd.concat([errors,pd.DataFrame([[mc.rho_prime,phys.E,phys.s,mc.N_e,MSE]], columns=errors.columns)], ignore_index=True)
    errors.to_csv(f"{PROJECT_ROOT}/results/sims/errors.csv", index=False)
    print(f"MSE:{MSE} with E: {phys.E}, with rho': {mc.rho_prime}, s: {phys.s}, traps: {mc.N_e}")
    print("done")
