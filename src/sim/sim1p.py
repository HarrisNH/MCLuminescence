import numpy as np
import pandas as pd
import functions as fc
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import hydra

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    configs = fc.initialize_runs(cfg)
    n_setups = len(configs.keys())

    for key in configs.keys():
        run_cfg = configs[key]
        mc = run_cfg.exp_type_fp
        phys = run_cfg.physics_fp
        if mc.exp_type == "background":
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups))
                x_ax = np.zeros((mc.steps, mc.sims,n_setups))
                Lum_sec = np.zeros((int(mc.x_lim / mc.bin_size), mc.sims,n_setups))
                electron_ratio = np.zeros((mc.steps, mc.sims,n_setups))
            x_ax, Lum, electron_ratio = fc.sim_bg(run_cfg, Lum, x_ax, electron_ratio, key, list(configs.keys()))
        elif mc.exp_type == "lab_TL":
            sim_results = pd.DataFrame(columns=["rho_prime","E","s","N_e","MSE"])
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups))
                x_ax = np.zeros((mc.steps, mc.sims,n_setups))
                Lum_sec = np.zeros((int(mc.x_lim / mc.bin_size), mc.sims,n_setups))
                electron_ratio = np.zeros((mc.steps, mc.sims,n_setups))
            MSE = fc.sim_lab_TL_residuals(run_cfg, Lum, x_ax, electron_ratio, key, list(configs.keys()))
            new_row = pd.DataFrame([{
                'rho_prime': mc.rho_prime,
                'E': phys.E,
                's': phys.s,
                'N_e': mc.N_e,
                'MSE': MSE
            }])
            
            sim_results = pd.concat([sim_results,new_row], ignore_index=True)
            
        elif mc.exp_type == "iso":
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups))
                x_ax = np.zeros((mc.steps, mc.sims,n_setups))
                Lum_sec = np.zeros((int(mc.x_lim / mc.bin_size), mc.sims,n_setups))
                electron_ratio = np.zeros((mc.steps, mc.sims,n_setups))

            for j in range(mc.sims):
                electrons, holes, e_max = fc.initialize_box(run_cfg)
                distances = fc.calc_distances(electrons, holes)
                min_distances, hole_index, = fc.min_distance(distances)
                lifetime = fc.lifetime_tunneling(run_cfg, min_distances)
                recombination = np.random.exponential(lifetime)
                timebin = np.zeros(mc.steps)

                if mc.distance_plot:
                    fc.distance_plot(run_cfg, min_distances)
                for i in range(mc.steps):
                    dt = np.min(recombination)-timebin[i - 1]
                    timebin[i] = timebin[i - 1] + dt 

                    #Handle recombination event
                    distances, electrons, holes,hole_index,min_distances,recombination,Lum[i,j,key] = fc.recomber(run_cfg,
                        recombination,electrons,holes,hole_index,distances,min_distances,timebin[i])
                    
                    electron_ratio[i,j,key] = electrons.shape[0]

                    if i % 1 == 0:
                        print(f"Step {i} of {mc.steps}, dt: {dt}")
                        print(f"Lum: {Lum[i,j,key]}, electrons: {electrons.shape[0]}")
                        
                        #distance_plot(run_cfg, min_distances)
                    if electrons.shape[0] == 0:
                        break
                time_passed = np.sum(timebin)
                print(f"Simulation {j} done and we simulated across {time_passed} seconds")
                print(f"Total luminescence for sim {j} is {np.sum(Lum[:i, j])}")
                print(f"Mean luminescence for sim {j} is {np.mean(Lum[:i, j])}")
                print(i / mc.steps)
                x_ax[:i,j,key] = timebin[:i]

        elif mc.exp_type == "TL":
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups))
                Times = int((mc.T_end-mc.T_start)/(mc.bin_size*mc.T_rate))
                x_ax_plot = np.linspace(mc.T_start, mc.T_end, Times+1)
                Lum_sec = np.zeros((len(x_ax_plot), mc.sims,n_setups))
                Temps = int((mc.T_end-mc.T_start)/(mc.bin_size))
                Lum_celsius = np.zeros((Temps+1, mc.sims,n_setups))
    
            for j in range(mc.sims):
                electrons, holes = fc.initialize_box(run_cfg)
                distances = fc.calc_distances(electrons, holes)
                min_distances, hole_index = fc.min_distance(distances)

                if mc.distance_plot and j == 0:
                    fc.distance_plot(run_cfg, min_distances)

                exp_duration = (mc.T_end-mc.T_start)/mc.T_rate
                timebin = np.zeros(mc.steps)

                #find lifetime of all electrons
                lifetime = fc.lifetime_thermal(run_cfg, min_distances,mc.T_start) #np.zeros(mc.electrons)+200 # 
                recombination = np.random.exponential(lifetime)               
                
                i=0
                while timebin[i-1] < exp_duration:
                    #Time step is decided by time until next recombination event or bin size and must be >0
                    dt = np.min((np.min(recombination)-timebin[i-1],2/mc.T_rate))-timebin[i-1] #max step
                    dt = np.max((dt,0))
                    timebin[i] = timebin[i - 1] + dt 
                    deltaT = (timebin[i])*mc.T_rate #udpate temperature based on step ######### remember to fix deltaT step as in Lab sim function

                    #Handle recombination event
                    distances, electrons, holes,hole_index,min_distances,recombination,Lum[i,j,key] = fc.recomber(
                        recombination,electrons,holes,hole_index,distances,min_distances,timebin[i],deltaT)
                    
                    lifetime = fc.lifetime_thermal(run_cfg, min_distances,deltaT)
                    recombination = np.random.exponential(lifetime)
                
                    if i % 150 == 0:
                        print(f"Step {i} of {mc.steps}")
                        #distance_plot(run_cfg, min_distances)
                    i+=1
                loop = [i,j,key]
                x_ax, Lum = fc.results_temp(run_cfg, Lum, timebin, electrons.shape[0], Lum_sec, Lum_celsius, loop)        
    if mc.exp_type == "lab":
        return MSE
    else: 
        return x_ax, Lum, electron_ratio, configs

if __name__ == "__main__":
    main()