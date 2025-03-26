import numpy as np
import functions as fc
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import hydra
import os

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    configs = fc.initialize_runs(cfg)
    n_setups = len(configs.keys())
    electron_ratio=0
    for key in configs.keys():
        run_cfg = configs[key]
        mc = run_cfg.exp_type_fp
        
        if mc.exp_type == "background":
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups))
                x_ax = np.zeros((mc.steps, mc.sims,n_setups))
                Lum_sec = np.zeros((int(mc.x_lim / mc.bin_size), mc.sims,n_setups))
                electron_ratio = np.zeros((mc.steps, mc.sims,n_setups))
            for j in range(mc.sims):
                electrons, holes,box_dim,e_max = fc.initialize_box_bg(run_cfg)
                e_timer = np.array([])
                dt_filling = fc.filling_time(run_cfg,electrons.shape[0],e_max)
                timebin = np.zeros(mc.steps)           
                distances = fc.calc_distances(electrons, holes)
                min_distances, hole_index = fc.min_distance(distances)
                
                #find lifetime of all electrons
                lifetime = fc.lifetime_tunneling(run_cfg, min_distances) #this needs to be updated for changing number of electrons 
                recombination = np.random.exponential(lifetime)     
                i=0
                t0 = 0 #check whether t0 is correct (how it is reset)
                exp_duration = dt_filling*mc.steps*0.6 #to ensure that array is big enough
                print_param = 100
                fill_counter,fill_attempts_counter = 0,0
                while timebin[i-1]<mc.duration:
                    #timenow = timebin[i-1]
                    #Time step is decided by time until next recombination or filling event
                    dt_recomb = np.min(recombination - e_timer[:recombination.shape[0]]) if recombination.size > 0 else dt_filling-t0 #check this timebin subtraction
                    dt = np.min((dt_recomb,dt_filling-t0))#2C max step
                    if dt == dt_filling-t0:
                        fill_attempts_counter += 1
                        t0 = 0
                        ratio_traps = (electrons.shape[0])/e_max
                        timebin[i] = timebin[i - 1] + dt
                        #if np.random.rand(1)+1 > ratio_traps:
                        fill_counter+=1
                        electrons, holes = fc.add_electron(run_cfg,box_dim,electrons,holes)
                        e_timer = np.append(e_timer,timebin[i])
                        distances = fc.recalc_distances(electrons, holes,distances)
                        min_distances, hole_index = fc.min_distance(distances)
                        lifetime = fc.lifetime_tunneling(run_cfg, min_distances)
                        recombination = np.random.exponential(lifetime)
                        dt_filling = fc.filling_time(run_cfg,electrons.shape[0],e_max)
                        #else:
                        #    pass
                    elif dt == dt_recomb:
                        dt = np.max((dt,0))
                        timebin[i] = timebin[i - 1] + dt
                        t0 += dt
                        #Handle recombination event
                        distances, electrons, holes,hole_index,min_distances,recombination,Lum[i,j,key],e_timer = fc.recomber(run_cfg,
                            recombination,electrons,holes,hole_index,distances,min_distances,timebin[i],e_timer)
                        
                        #Recalculate distances for electrons that shared the recombined hole
                        lifetime = fc.lifetime_tunneling(run_cfg, min_distances)
                        recombination = np.random.exponential(lifetime)
                    electron_ratio[i,j,key] = electrons.shape[0]/e_max

                    if i%print_param  == 0:
                        # Pause briefly (if needed), then move the cursor up two lines.
                        # \033[F moves the cursor up one line.
                        if i!= 0:
                            print("\033[F\033[F\033[F\033[F\033[F\033[F", end="")  # Moves the cursor up two lines

                        # Print the two lines initially.
                        print(f"Step {i+1} of {mc.steps} | Sim {j+1} of {mc.sims} | Experiment {key} of {list(configs.keys())}")
                        print(f"Trap filling ratio is {electron_ratio[i,j,key]:.2f}", flush=True)
                        print(f"Lum in last {print_param} steps is {np.sum(Lum[i-print_param:i,j,key])}", flush=True)
                        print(f"{timebin[i]/(3600*24*365*1000):.2f} thousand years passed of max {exp_duration/(3600*24*365*1000):.2f}", flush=True)
                        print(f"Time passed in last {print_param} steps: {(timebin[i] - timebin[i-print_param])/(3600*24*365*1000):.2f} thousand years", flush=True)
                        print(f"Filling events in last {print_param} steps: {fill_counter} with {fill_attempts_counter} attempts", flush=True)
                        fill_counter, fill_attempts_counter= 0,0
                    if electron_ratio[i,j,key] > 0.98:
                        break
                    i+=1
                    
                        #distance_plot(run_cfg, min_distances)
                print(f"Simulation {j} done and we simulated across {timebin[i-1]/(3600*24*365*1000)} thousand years")

                x_ax[:,j,key] = timebin
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
                #for idx, x_t in enumerate(x_ax):
                #    Lum_index = np.where((timebin[:i] < (x_t + mc.bin_size)) & (timebin[:i] > x_t))
                #    Lum_sec[idx, j,key] = np.sum(Lum[:i, j,key][Lum_index])
                print(f"Total luminescence for sim {j} is {np.sum(Lum[:i, j])}")
                print(f"Mean luminescence for sim {j} is {np.mean(Lum[:i, j])}")
                print(i / mc.steps)
                x_ax[:i,j,key] = timebin[:i]

        elif mc.exp_type == "TL":
            #Setup
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups)) #steps is just used to ensure Lum container has enough rows
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
                    deltaT = (timebin[i])*mc.T_rate #udpate temperature based on step

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
    return x_ax, Lum, electron_ratio

if __name__ == "__main__":
    main()