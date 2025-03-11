import numpy as np
import matplotlib.pyplot as plt
from paths import CONFIG_DIR,PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import hydra

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
                    # Convert ListConfig to a list.
                    lst = list(v)
                    # Check that the list length divides max_length evenly.
                    if max_length % len(lst) != 0:
                        raise ValueError(
                            f"max_length ({max_length}) is not divisible "
                            f"by len({sub_name}.{k}) ({len(lst)})"
                        )
                    # Extend the list by repeating it so that its length equals max_length.
                    extended_list = lst * (max_length // len(lst))
                    # Pick the i-th element for this run.
                    sub_run[k] = extended_list[i]
                else:
                    # For non-list values, use the same value for every run.
                    sub_run[k] = v
            run_dict[sub_name] = sub_run
        runs[i] = run_dict

    return runs
def initialize_box(cfg):
    '''
    Initialize the electrons and holes in the box centered at the origin
    '''
    mc = cfg["exp_type_fp"]
    e = mc.electrons
    h = mc.holes
    box_l,box_w,box_h = mc.box_l,mc.box_w,mc.box_h

    b_factor = 1.2 # boundary factor
    box_l_b= box_l*b_factor
    box_w_b = box_w*b_factor
    box_h_b = box_h*b_factor
    box_volume = box_l*box_w*box_h
    box_volume_b = box_l_b*box_w_b*box_h_b
    scaling = np.array([box_l, box_w, box_h])
    electrons = (np.random.rand(e,3))*scaling # (l,w,h) with mean = 0, uniform distrubted in the box
    holes_density = h/(box_l*box_w*box_h)
    b_volume = box_volume_b - box_volume
    holes_boundary_n = int(holes_density*b_volume)
    holes_n = h + holes_boundary_n
    scaling_b = np.array([box_l_b, box_w_b, box_h_b])
    holes =  (np.random.rand(holes_n,3))*scaling_b

    ### does nearest neighhbor distribution is that correct - okay maybe 20%percent
    return electrons,holes

def calc_distances(electrons: np.array,holes: np.array):
    '''
    Calculate the distances between all electrons and holes, each row is one electron and each column is one hole
    '''
    distances = np.linalg.norm(electrons[:, np.newaxis, :] - holes[np.newaxis, :, :], axis=2)
    min_dist = np.min(distances,axis=1)
    min_indices = np.argmin(distances, axis=1)
    unique, counts = np.unique(min_indices, return_counts=True)
    return min_dist,min_indices

def theoretical_nearest_neighbor_dist(cfg,r):
    mc = cfg["exp_type_fp"]
    box_l,box_w,box_h = mc.box_l,mc.box_w,mc.box_h
    rho = mc.holes/(box_l*box_w*box_h)
    NND = np.exp(-4/3*np.pi*rho*r**3)*4*np.pi*rho*r**2 #nearest neighbour distribution
    return NND


def probability_thermal(cfg,distances):
    '''
    Calculate the probability of tunneling for each electron-hole pair
    Formula: b*exp(-(etha+alpha*distances))
    '''
    mc = cfg["exp_type_fp"]
    phys = cfg["physics_fp"]
    xi = phys.E/(phys.k_b*(mc.T_start+273.15))
    rate = phys.b*np.exp(-(xi+phys.alpha*distances))
    return rate

def distance_plot(cfg,distances):
    r = np.linspace(0, 3, 100)
    dist_theoretical = theoretical_nearest_neighbor_dist(cfg, r)
    plt.hist(distances[:], bins=30,density = True)
    #plt.hist(np.mean(distances,axis=1), bins=30,density = True)
    plt.plot(r, dist_theoretical)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg):
    '''
    Main function to run the simulation
    '''
    configs = initialize_runs(cfg)
    for k in configs.keys():
        mc = configs[k]["exp_type_fp"]
        phys = configs[k]["physics_fp"]
        if mc["exp_type"] == "iso":
            Lum = np.zeros((mc["steps"],mc["sims"]))
            x_ax = np.linspace(0,mc["x_lim"]-1,int(mc["x_lim"]/mc["bin_size"]))
            Lum_sec = np.zeros((int(mc["x_lim"]/mc["bin_size"]),mc["sims"]))
            for j in range(mc["sims"]):
                electrons, holes = initialize_box(configs[k])
                distances,hole_index = calc_distances(electrons,holes)
                rate = probability_thermal(cfg,distances)
                timebin = np.zeros(mc.steps)
                if mc.distance_plot:
                    distance_plot(cfg,distances)

                for i in range(mc.steps):
                    dt = np.min(1/rate)/100 #min of lifetime divide by 10 to choose timestep
                    timebin[i] = timebin[i-1] + dt
                    recombination = np.random.binomial(1,rate[:,]*dt).astype(bool)
                    electrons = electrons[~recombination]
                    ## need to track what holes have recombined 
                    # (we assumme no electrons ever recombine with same hole at same timestep)
                    holes = np.delete(holes,hole_index[recombination],0)
                    distances,hole_index = calc_distances(electrons,holes)
                    rate = probability_thermal(cfg,distances)
                    Lum[i,j] = np.sum(recombination)  #be aware that mc.bin_size affects 
                    #what numerator should be as it scales Lum to be Lum per 1sec for example when dt is numerator (i think)

                    if len(rate)/mc.electrons < 0.10: #only 5% of electrons left
                        break
                    if i % 1000 == 0:
                        print(f"Step {i} of {mc.steps}")
                time_passed = sum(timebin)
                print(f"Simulation {j} done and we simulated across {time_passed} seconds")
                #
                #time_in_plot = timebin[:i][timebin[:i]<mc.x_lim]

                for k,x_t in enumerate(x_ax):
                    Lum_index = np.where((timebin[:i]<x_t+mc.bin_size) & (timebin[:i]>x_t))
                    Lum_sec[k,j] = np.sum(Lum[:i,j][Lum_index])
                print(f"Total luminesce for sim {j} is {np.sum(Lum[:i,j])}")
                print(f"Mean luminesce for sim {j} is {np.mean(Lum[:i,j])}")
                print(i/mc.steps)

        elif mc.exp_type == "TL":
            print("Under construction")
            plt.plot(x_ax,Lum_sec[:,j])
            plt.xlim(0,mc.x_lim)
            #plt.xscale('log')
    return x_ax,Lum_sec
    



if __name__ == "__main__":
    main()
