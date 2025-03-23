import numpy as np
import matplotlib.pyplot as plt
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import hydra
import os

def initialize_runs_old(cfg):
    
    configs = {
        "exp_type_fp": cfg.exp_type_fp,
        "physics_fp": cfg.physics_fp,
        "setup": cfg.setup
    }
    # Get maximum length of any list in all dictionaries
    max_length = max(
        len(v)
        for sub_dict in configs.values()
        for v in sub_dict.values()
        if isinstance(v, ListConfig)
    )
    # Extend all lists to have the same length
    for sub_name, sub_dict in configs.items():
        for k, v in sub_dict.items():
            if isinstance(v, ListConfig):
                v = list(v)
                if max_length % len(v) != 0:
                    raise ValueError(
                        f"max_length ({max_length}) is not divisible "
                        f"by len({sub_name}.{k}) ({len(v)})"
                    )
                sub_dict[k] = v * (max_length // len(v))
    return configs

def initialize_runs(cfg):
    # Build the original dictionary of dictionaries
    configs = {
        "exp_type_fp": cfg.exp_type_fp,
        "physics_fp": cfg.physics_fp,
        "setup": cfg.setup
    }

    # Determine the maximum length among all lists in the nested dictionaries.
    max_length = max(
        len(v)
        for sub_dict in configs.values()
        for v in sub_dict.values()
        if isinstance(v, ListConfig)
    )

    # Build a new dictionary where keys are run indices 0, 1, ... max_length-1.
    runs = {}
    for i in range(max_length):
        run_dict = {}
        # For each category (exp_type_fp, physics_fp, setup) create a sub-dictionary.
        for sub_name, sub_dict in configs.items():
            sub_run = {}
            for k, v in sub_dict.items():
                if isinstance(v, ListConfig):
                    lst = list(v)
                    if max_length % len(lst) != 0:
                        raise ValueError(
                            f"max_length ({max_length}) is not divisible "
                            f"by len({sub_name}.{k}) ({len(lst)})"
                        )
                    # Extend the list by repeating it so that its length equals max_length.
                    extended_list = lst * (max_length // len(lst))
                    sub_run[k] = extended_list[i]
                else:
                    sub_run[k] = v
            # Convert the sub-run dictionary into an OmegaConf object
            run_dict[sub_name] = OmegaConf.create(sub_run)
        # Convert the run dictionary into an OmegaConf object so that dot access works.
        runs[i] = OmegaConf.create(run_dict)
    return runs

def initialize_box(cfg):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    mc = cfg.exp_type_fp
    e = mc.electrons
    h = mc.holes
    box_l, box_w, box_h = mc.box_l, mc.box_w, mc.box_h

    b_factor = mc.boundary_factor  # boundary factor
    box_l_b = box_l * b_factor
    box_w_b = box_w * b_factor
    box_h_b = box_h * b_factor
    box_volume = box_l * box_w * box_h
    box_volume_b = box_l_b * box_w_b * box_h_b
    scaling = np.array([box_l, box_w, box_h])
    electrons = (np.random.rand(e, 3)) * scaling
    holes_density = h / (box_l * box_w * box_h)
    b_volume = box_volume_b - box_volume
    holes_boundary_n = int(holes_density * b_volume)
    holes_n = h + holes_boundary_n
    scaling_b = np.array([box_l_b, box_w_b, box_h_b])
    holes = (np.random.rand(holes_n, 3)) * scaling_b
    print(f"Initialized {e} electrons and {holes_n} holes (with boundary)")
    return electrons, holes

def calc_distances(electrons: np.array, holes: np.array):
    """
    Calculate the distances between all electrons and holes, each row is one electron and each column is one hole.
    """
    distances = np.linalg.norm(electrons[:, np.newaxis, :] - holes[np.newaxis, :, :], axis=2)
    min_dist = np.min(distances, axis=1)
    min_indices = np.argmin(distances, axis=1)
    return min_dist, min_indices,distances

def theoretical_nearest_neighbor_dist(cfg, r):
    mc = cfg.exp_type_fp
    box_l, box_w, box_h = mc.box_l, mc.box_w, mc.box_h
    rho = mc.holes / (box_l * box_w * box_h)
    NND = np.exp(-4/3 * np.pi * rho * r**3) * 4 * np.pi * rho * r**2  # densitydistribution
    #NND_log = (-4/3 * np.pi * rho * r**3)+np.log(4 * np.pi * rho)+ 2*np.log(r)  # log densitydistribution
    return NND

def lifetime_thermal(cfg, distances,temp=0):
    """
    Calculate the probability of tunneling for each electron-hole pair.
    Formula: b * exp(-(etha + alpha * distances))
    """
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    xi = phys.E / (phys.k_b * (mc.T_start + 273.15+temp))
    rate = phys.b * np.exp(-(xi + phys.alpha * distances))
    return 1/rate

def distance_plot(cfg, distances):
    r = np.linspace(0, np.max(distances), 100)
    dist_theoretical = theoretical_nearest_neighbor_dist(cfg, r)
    if cfg.exp_type_fp.distance_plot:
        plt.hist((distances), bins=100, density=True)
        plt.plot(r, dist_theoretical)
        plt.tight_layout()
        plt.show()

def theoretical_lifetime_density(lifetime,time_sec):
    return 1/lifetime*np.exp(-time_sec/lifetime)

def lifetime_plot(cfg, Lum_sec, time_sec = 0, lifetime = 200):
    mc = cfg.exp_type_fp
    #density_theoretical = theoretical_lifetime_density(lifetime,time_sec)
    x_ax = np.arange(0,len(Lum_sec),1)
    plt.plot(x_ax,Lum_sec/sum(Lum_sec),color = "red")
    #plt.plot(time_sec, density_theoretical,color = "black")
    #plt.xlim(0,1000)
    #plt.ylim(0,0.1)
    #plt.savefig(os.path.join(PROJECT_ROOT, f"results/figs_life/lifetime_plot.png"))
    plt.show()
    print("done")


def recalc_distances(hole_index,distances_small,electrons_new_d,min_distances):
    """
    Find the new min distances and index of the electrons who lost their closest hole after another recombined.
    """
 
    #Recalculate the new min distance for the electron that shared the hole
    new_min_dist = np.min(distances_small, axis=1)
    new_min_idx = np.argmin(distances_small, axis=1)

    #Update the minimum distances and hole index
    new_min_dist
    hole_index[electrons_new_d] = new_min_idx
    return new_min_dist,hole_index

def remove_electrons(distances, electrons,holes,hole_index,min_distances, electrons_new_d,recombination,recombination_idx):
        #Remove the electron and hole that recombined
    electrons = np.delete(electrons, recombination_idx, 0)      #delete electron
    holes = np.delete(holes, hole_index[recombination_idx], 0)  #delete hole
    distances = np.delete(distances, recombination_idx, 0)      #delete electron row from distances
    distances = np.delete(distances, hole_index[recombination_idx], 1)  #delete hole column from distances
    min_distances = np.delete(min_distances, recombination_idx) #delete min distance of removed electron
    hole_index = np.delete(hole_index, recombination_idx)   #delete index of nearest hole
    recombination = np.delete(recombination, recombination_idx)
   
    # For holes that have index changed (bcs they were after the removed hole), update their index
    # This should also be done for the indices showing which electrons shold havetheir distances updated
    for i in range(len(recombination_idx)):
        hole_index[hole_index >= recombination_idx[i]] -= 1
        recombination_idx[recombination_idx >= recombination_idx[i]] -= 1
        electrons_new_d[electrons_new_d >= recombination_idx[i]] -= 1
    return distances, electrons, holes,hole_index,min_distances,electrons_new_d,recombination

def electrons_new_distances(hole_index, recombination_idx):
    holes_that_recombined = hole_index[recombination_idx]  # shape could be (2,) etc.

    # Boolean mask: does hole_index[i] match any of the holes_that_recombined?
    mask_hole_match = np.isin(hole_index, holes_that_recombined)

    # Boolean mask: is electron index i in recombination_idx?
    mask_not_recombining_e = ~np.isin(np.arange(len(hole_index)), recombination_idx)

    # Combine the two conditions:
    electrons_new_d = np.where(mask_hole_match & mask_not_recombining_e)[0]
    return electrons_new_d

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    configs = initialize_runs(cfg)
    n_setups = len(configs.keys())

    for key in configs.keys():
        run_cfg = configs[key]
        mc = run_cfg.exp_type_fp

        if mc.exp_type == "iso":
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups))
                x_ax = np.linspace(0, mc.x_lim - 1, int(mc.x_lim / mc.bin_size))
                Lum_sec = np.zeros((int(mc.x_lim / mc.bin_size), mc.sims,n_setups))
            for j in range(mc.sims):
                electrons, holes = initialize_box(run_cfg)
                min_distances, hole_index,distances = calc_distances(electrons, holes)
                rate = lifetime_thermal(run_cfg, min_distances)
                timebin = np.zeros(mc.steps)
                if mc.distance_plot:
                    distance_plot(run_cfg, min_distances)
                for i in range(mc.steps):
                    recombination = np.random.exponential(1/rate)
                    dt = np.min(recombination)
                    recombination_idx = np.argmin(recombination)
                    timebin[i] = timebin[i - 1] + dt 
                    #Check whether recalculation has to happen before deleting the electron and hole
                    electrons_new_d = np.where(
                        (hole_index == hole_index[recombination_idx]) & (np.arange(len(hole_index)) != recombination_idx)
                    )[0]#electrons to have new distance
                    distances, electrons, hole_index,min_distances,electrons_new_d = remove_electrons(distances, electrons,holes,hole_index,recombination_idx,min_distances,electrons_new_d)
                    if electrons_new_d.shape[0] > 1:  
                        min_distances,hole_index = recalc_distances(hole_index,distances[electrons_new_d],electrons_new_d,min_distances)
                    
                    #distances, hole_index = calc_distances(electrons, holes)
                    rate = lifetime_thermal(run_cfg, min_distances)
                    Lum[i, j,key] = np.sum(recombination)
                    if len(rate) / mc.electrons < 0.10:
                        break
                    if i % 1000 == 0:
                        print(f"Step {i} of {mc.steps}")
                        #distance_plot(run_cfg, min_distances)
                time_passed = np.sum(timebin)
                print(f"Simulation {j} done and we simulated across {time_passed} seconds")
                for idx, x_t in enumerate(x_ax):
                    Lum_index = np.where((timebin[:i] < (x_t + mc.bin_size)) & (timebin[:i] > x_t))
                    Lum_sec[idx, j,key] = np.sum(Lum[:i, j,key][Lum_index])
                print(f"Total luminescence for sim {j} is {np.sum(Lum[:i, j])}")
                print(f"Mean luminescence for sim {j} is {np.mean(Lum[:i, j])}")
                print(i / mc.steps)
        elif mc.exp_type == "TL":
            #Setup
            if key == 0:
                Lum = np.zeros((mc.steps, mc.sims,n_setups)) #steps is just used to ensure Lum container has enough rows
                Times = int((mc.T_end-mc.T_start)/(mc.bin_size*mc.T_rate))
                Temps = int((mc.T_end-mc.T_start)/(mc.bin_size))
                x_ax_plot = np.linspace(mc.T_start, mc.T_end, Times+1)
                Lum_sec = np.zeros((len(x_ax_plot), mc.sims,n_setups))

                Lum_celsius = np.zeros((Temps, mc.sims,n_setups))
            for j in range(mc.sims):
                electrons, holes = initialize_box(run_cfg)
                min_distances, hole_index,distances = calc_distances(electrons, holes)
                lifetime = lifetime_thermal(run_cfg, min_distances,mc.T_start) #np.zeros(mc.electrons)+200 # 
                exp_duration = (mc.T_end-mc.T_start)/mc.T_rate
                timebin = np.zeros(mc.steps)

                #find lifetime of all electrons
                recombination = np.random.exponential(lifetime)               
                if mc.distance_plot: #plot what distance to holes are
                    distance_plot(run_cfg, min_distances)
                i=0


                while timebin[i-1] < exp_duration:
                    #Time step is decided by time until next recombination event or bin size
                    dt = np.min((np.min(recombination)-timebin[i-1],5/mc.T_rate))#5C max step
                    #Time step not less that zero
                    dt = np.max((dt,0))
                    if dt != 1.0 and dt != 0:
                        print(dt)
                    timebin[i] = timebin[i - 1] + dt 
                    deltaT = (timebin[i])*mc.T_rate #udpate temperature based on step

                    #Check what electrons have exceeded their lifetimes
                    recombination_idx = np.where((timebin[i] >= recombination))[0]

                    #Check what electrons need to have recalculated distances
                    electrons_new_d = electrons_new_distances(hole_index, recombination_idx)

                    #remove electrons and holes that recombined and ensure electrons that need new distance actually has correct index
                    distances, electrons, holes,hole_index,min_distances,electrons_new_d,recombination = remove_electrons(
                        distances, electrons,holes,hole_index,min_distances,electrons_new_d,recombination,recombination_idx)
                    if electrons.shape[0]/ mc.electrons == 0:
                        #distance_plot(run_cfg, min_distances)
                        break
                    
                    Lum[i, j,key] = len(recombination_idx)
                    if electrons_new_d.shape[0] > 1:  
                        new_min_distances,hole_index = recalc_distances(hole_index,distances[electrons_new_d],electrons_new_d,min_distances[electrons_new_d])
                        min_distances[electrons_new_d] = new_min_distances
                    lifetime = lifetime_thermal(run_cfg, min_distances,mc.T_start+deltaT) #lifetime[:len(new_min_distances)]#
                    recombination = np.random.exponential(lifetime)
                
                    if i % 1000 == 0:
                        print(f"Step {i} of {mc.steps}")
                        distance_plot(run_cfg, min_distances)
                    i+=1
                time_passed = int(timebin[i-1])
                x_ax_time = np.arange(0,time_passed,mc.bin_size)
                x_ax_celsius = np.arange(0,time_passed*mc.T_rate,mc.bin_size)
                print(f"Simulation {j} done and we simulated across {time_passed} seconds with temp_increase of {mc.T_rate}/s")
                print(f"The ratio of electrons that recombined is {1-electrons.shape[0]/mc.electrons}")
                for idx, x_t in enumerate(x_ax_time): #This loop bins luminecscense data for each temperature (decided by bin size)
                    Lum_index_sec = np.where((timebin[:i] <= (x_t + mc.bin_size)) & (timebin[:i] > x_t))
                    Lum_sec[idx, j,key] = np.sum(Lum[:i, j,key][Lum_index_sec])
                for idx, x_t in enumerate(x_ax_celsius):
                    Lum_index_cel = np.where((timebin[:i] <= (x_t + mc.bin_size)/mc.T_rate) & (timebin[:i] > x_t/mc.T_rate))
                    Lum_celsius[idx,j,key] = np.sum(Lum[:i, j,key][Lum_index_cel])
                print(f"Total luminescence for sim {j} is {np.sum(Lum[:i, j,key])}")
                print(f"Mean luminescence per second for sim {j} is {np.mean(Lum_sec[:, j,key])}")
                print(i / mc.steps)
                #lifetime_plot(run_cfg, Lum_sec[:time_passed,j,key])
    return x_ax_celsius, Lum_celsius

if __name__ == "__main__":
    main()