from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import numpy as np
import matplotlib.pyplot as plt

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


def rho_func(cfg):
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    rho = mc.rho_prime*(3/(4*np.pi)*phys.alpha**3)
    return rho
def initialize_box_bg(cfg):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    mc = cfg.exp_type_fp
    e = mc.electrons
    e_max = mc.N_e
    box_l, box_w, box_h = mc.d,mc.d,mc.d
    box_volume = box_l*box_w*box_h
    rho = rho_func(cfg)
    h = int(rho*box_volume) #this doesnt have to be true - but holes have to match with rho

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
    return electrons, holes, [box_l,box_w,box_h], e_max

def initialize_box(cfg):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    mc = cfg.exp_type_fp
    e = int(mc.electrons)
    rho = rho_func(cfg) #using rho prime and alpha
    
    box_l, box_w, box_h = mc.d,mc.d,mc.d
    box_volume = box_l*box_w*box_h
    h = int(rho*box_volume)

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
    return electrons, holes, [box_l,box_w,box_h]

def calc_distances(electrons: np.array, holes: np.array):
    """
    Calculate the distances between all electrons and holes, each row is one electron and each column is one hole.
    """
    distances = np.linalg.norm(electrons[:, np.newaxis, :] - holes[np.newaxis, :, :], axis=2)
    return distances

def min_distance(distances):
    """
    Find the minimum distance between each electron and hole.
    """
    min_dist = np.min(distances, axis=1)
    min_indices = np.argmin(distances, axis=1)
    return min_dist, min_indices

def theoretical_nearest_neighbor_dist(cfg, r):
    mc = cfg.exp_type_fp
    box_l, box_w, box_h = mc.box_l, mc.box_w, mc.box_h
    rho = mc.rho_prime #density of holes in m^-3
    NND = np.exp(-4/3 * np.pi * rho * r**3) * 4 * np.pi * rho * r**2  # densitydistribution of holes
    return NND

def lifetime_thermal(cfg, distances,temp=0):
    """
    Calculate the probability of tunneling for each electron-hole pair.
    Formula: b * exp(-(etha + alpha * distances))
    Temp input is in celsuis, then converted to kelvin inside function
    """
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    xi = phys.E / (phys.k_b * (mc.T_start + 273.15+temp))
    rate = phys.b * np.exp(-(xi + phys.alpha * distances))
    return 1/rate

def lifetime_tunneling(cfg,distances):
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    lifetime = 1/phys.s*np.exp(phys.alpha*distances)
    return lifetime

def distance_plot(cfg, distances):
    # Convert distances from metres to nanometres for plotting
    distances_nm = distances * 1e9
    # Create an array of r values in metres and then convert to nm for x-axis plotting
    r_m = np.linspace(0, np.max(distances)+2e-9, 100)
    r_nm = r_m * 1e9
    # Calculate the theoretical nearest-neighbour distribution using r in metres
    dist_theoretical = theoretical_nearest_neighbor_dist(cfg, r_m)/1e9 #down to nm
    if cfg.exp_type_fp.distance_plot:
        plt.hist(distances_nm, bins=30, density=True, alpha=0.7)
        plt.plot(r_nm, dist_theoretical, color="red")
        plt.xlabel("Distance (nm)")
        plt.ylabel("Probability density")
        plt.tight_layout()
        plt.show()
        print("plot")

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

def recalc_distances(electrons,holes,distances):
    """
    Recalculate distances from the latest electron/hole to all holes/electrons.
    """
    last_electron = electrons[-1]  # Select only the last electron
    last_hole = holes[-1]  # Select only the last hole
    distances_last_e = np.linalg.norm(last_electron - holes, axis=1)[:-1]
    distances_last_h = np.linalg.norm(electrons - last_hole, axis=1)
    distances = np.vstack([distances, distances_last_e])
    distances = np.hstack([distances, distances_last_h[:, np.newaxis]])

    return distances

def recalc_min_distances(hole_index,distances_small,electrons_new_d,min_distances):
    """
    Find the new min distances and index of the electrons who lost their closest hole after another recombined.
    """
 
    #Recalculate the new min distance for the electron that shared the hole
    new_min_dist = np.min(distances_small, axis=1)
    new_min_idx = np.argmin(distances_small, axis=1)

    #Update the minimum distances and hole index
    hole_index[electrons_new_d] = new_min_idx
    return new_min_dist,hole_index


def remove_electrons(distances, electrons,holes,hole_index,min_distances, electrons_new_d,recombination,recombination_idx,e_timer):
        #Remove the electron and hole that recombined
    electrons = np.delete(electrons, recombination_idx, 0)      #delete electron
    holes = np.delete(holes, hole_index[recombination_idx], 0)  #delete hole
    distances = np.delete(distances, recombination_idx, 0)      #delete electron row from distances
    distances = np.delete(distances, hole_index[recombination_idx], 1)  #delete hole column from distances
    min_distances = np.delete(min_distances, recombination_idx) #delete min distance of removed electron
    hole_index = np.delete(hole_index, recombination_idx)   #delete index of nearest hole
    recombination = np.delete(recombination, recombination_idx)
    e_timer = np.delete(e_timer, recombination_idx)
    # For holes that have index changed (bcs they were after the removed hole), update their index
    # This should also be done for the indices showing which electrons should have their distances updated
    for i in range(len(recombination_idx)):
        hole_index[hole_index >= recombination_idx[i]] -= 1
        recombination_idx[recombination_idx >= recombination_idx[i]] -= 1
        electrons_new_d[electrons_new_d >= recombination_idx[i]] -= 1
    return distances, electrons, holes, hole_index, min_distances, electrons_new_d, recombination,e_timer

def electrons_new_distances(hole_index, recombination_idx):
    holes_that_recombined = hole_index[recombination_idx]  # shape could be (2,) etc.

    # Boolean mask: does hole_index[i] match any of the holes_that_recombined?
    mask_hole_match = np.isin(hole_index, holes_that_recombined)

    # Boolean mask: is electron index i in recombination_idx?
    mask_not_recombining_e = ~np.isin(np.arange(len(hole_index)), recombination_idx)

    # Combine the two conditions:
    electrons_new_d = np.where(mask_hole_match & mask_not_recombining_e)[0]
    return electrons_new_d

def filling_time(cfg,electrons,e_max):
    """
    Calculate the filling rate of the electrons in the box
    """
    phys = cfg.physics_fp
    mc = cfg.exp_type_fp
    N = e_max #check whether this should always be zero
    lifetime = phys.D0/phys.D*1/(N-electrons)
    return lifetime

def add_electron(cfg,box_dim,electrons,holes):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    mc = cfg.exp_type_fp
    b_factor = mc.boundary_factor  # boundary factor
    box_l, box_w, box_h = box_dim
    box_l_b, box_w_b, box_h_b = box_dim[0] * b_factor,box_dim[1] * b_factor,box_dim[2] * b_factor

    scaling = np.array([box_l, box_w, box_h])
    new_electron = (np.random.rand(1, 3)) * scaling
    electrons = np.vstack([electrons, new_electron])
    scaling_b = np.array([box_l_b, box_w_b, box_h_b])
    new_hole = (np.random.rand(1, 3)) * scaling_b
    holes = np.vstack([holes, new_hole])
    return electrons, holes

def recomber(cfg,recombination,electrons,holes,hole_index,distances,min_distances,time,e_timer):
    """
    Recalculate the recombination time and remove the electron and hole that recombined
    """
    #Check what electrons have exceeded their lifetimes
    recombination_idx = np.where((time >= recombination))[0]
    hole_index[recombination_idx]
    #Check what electrons need to have recalculated distances
    electrons_new_d = electrons_new_distances(hole_index, recombination_idx)
    
    #remove electrons and holes that recombined and ensure electrons that need new distance actually has correct index
    distances, electrons, holes,hole_index,min_distances,electrons_new_d,recombination,e_timer = remove_electrons(
                        distances, electrons,holes,hole_index,min_distances,electrons_new_d,recombination,recombination_idx,e_timer)
    #Calculate luminiscence
    Lum = len(recombination_idx)

    if electrons_new_d.shape[0] > 0:  
        min_distances[electrons_new_d], hole_index[electrons_new_d] = min_distance(distances[electrons_new_d])
        lifetime = lifetime_tunneling(cfg, min_distances[electrons_new_d])
        recombination[electrons_new_d] = np.random.exponential(lifetime)

    return distances, electrons, holes,hole_index,min_distances,recombination,Lum,e_timer



def analy_TL_iso(cfg, timebin,i):
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    ns = mc.electrons*np.exp(-mc.rho_prime*np.log(1.8*phys.s*timebin)**3)   
    return ns                 

               
def results_temp(run_cfg, Lum, timebin, electrons, Lum_sec, Lum_celsius, loop):
    i,j,key = loop
    mc = run_cfg.exp_type_fp
    time_passed = int(timebin[i-2])
    x_ax_time = np.arange(0,time_passed,mc.bin_size)
    x_ax_celsius = np.arange(0,time_passed*mc.T_rate,mc.bin_size)
    print(f"Simulation {j} done and we simulated across {time_passed} seconds with temp_increase of {mc.T_rate}/s")
    print(f"The ratio of electrons that recombined is {1-electrons/mc.electrons}")
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
