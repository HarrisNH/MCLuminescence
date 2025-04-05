from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import numpy as np
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT
import pandas as pd

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

def initialize_box_bg(cfg,e_ratio_start=False):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    phys = cfg.physics_fp
    mc = cfg.exp_type_fp
    if not e_ratio_start:
        e = mc.electrons
    else: e = int(mc.N_e*e_ratio_start)

    rho = rho_func(cfg)
    h = int(mc.holes)
    d = (h/rho)**(1/3)
    box_l, box_w, box_h = d, d, d
    box_volume = box_l*box_w*box_h
    #int(rho*box_volume) #this doesnt have to be true - but holes have to match with rho
    if h<e:
        print(f"Not enough holes for the electrons: holes = {h} and electrons = {e}")
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
    return electrons, holes, [box_l,box_w,box_h]

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

def min_distance(distances,distances_all=0,electrons_new_d=0):
    """
    Find the minimum distance between each electron and hole.
    """
    try:
        min_dist = np.min(distances, axis=1)
        min_indices = np.argmin(distances, axis=1)
    except Exception as e:
        min_dist = np.zeros(distances.shape[0])+1e20
        min_indices = 0
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
    xi = phys.E / (phys.k_b * (temp))
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
    while recombination_idx.shape[0] > 0 and hole_index.shape[0] > 0:
        # Update index values before we delete
        hole_index[hole_index > hole_index[recombination_idx[0]]] -= 1 ### think deeply (when -1 values then never recombine (but if two electrons share same hole then the other electron needs to get a new distance. THINK THINK))
        electrons_new_d[electrons_new_d > recombination_idx[0]] -= 1

        electrons = np.delete(electrons, recombination_idx[0], 0)      #delete electron
        holes = np.delete(holes, hole_index[recombination_idx[0]], 0)  #delete hole
        distances = np.delete(distances, recombination_idx[0], 0)      #delete electron row from distances
        distances = np.delete(distances, hole_index[recombination_idx[0]], 1)  #delete hole column from distances
        min_distances = np.delete(min_distances, recombination_idx[0]) #delete min distance of removed electron
        hole_index = np.delete(hole_index, recombination_idx[0])   #delete index of nearest hole
        recombination = np.delete(recombination, recombination_idx[0])
        e_timer = np.delete(e_timer, recombination_idx[0])
        # For holes that have index changed (bcs they were after the removed hole), update their index
        # This should also be done for the indices showing which electrons should have their distances updated
        recombination_idx[recombination_idx >= recombination_idx[0]] -= 1 #### i just made a change here
        recombination_idx = np.delete(recombination_idx, 0)  #delete the first recombination index

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

def filling_time(cfg,electrons,e_max,D=0):
    """
    Calculate the filling rate of the electrons in the box
    """
    phys = cfg.physics_fp
    mc = cfg.exp_type_fp
    N = int(e_max)
    if electrons==N or D == 0:
        lifetime = 1e20
    else: lifetime = phys.D0/D*1/(N-electrons)
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

def reduce_recombination_idx(hole_index, recombination_idx):
    '''
    This functions ensures that if two recombining electrons share the same hole, only one is reocmbined. 
    '''
    seen = set()
    filtered_idx = []
    for idx in recombination_idx:
        val = hole_index[idx]
        if val not in seen:
            seen.add(val)
            filtered_idx.append(idx)
    return np.array(filtered_idx)

def recomber(cfg,recombination,electrons,holes,hole_index,distances,min_distances,time,e_timer):
    """
    Recalculate the recombination time and remove the electron and hole that recombined
    """
    #Check what electrons have exceeded their lifetimes taking into account when they got filled
    recombination_idx = np.where((time >= recombination+e_timer))[0] ##wwhHAAIIITT is this correct?
    if recombination_idx.shape[0] > 1:
        recombination_idx = reduce_recombination_idx(hole_index, recombination_idx)
        

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
    #if holes.shape[0] == 0: #if no holes set next recombination very far out since there can be no recombination
    #    recombination = recombination+1e20
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




def sim_bg(run_cfg, Lum, x_ax, electron_ratio,key,exp_list):
            mc = run_cfg.exp_type_fp
            for j in range(mc.sims):
                electrons, holes,box_dim,e_max = initialize_box_bg(run_cfg)
                e_timer = np.array([])
                dt_filling = filling_time(run_cfg,electrons.shape[0],e_max,D)
                timebin = np.zeros(mc.steps)           
                distances = calc_distances(electrons, holes)
                if distances.size != 0:
                    min_distances, hole_index = min_distance(distances)
                
                    #find lifetime of all electrons
                    lifetime = lifetime_tunneling(run_cfg, min_distances) #this needs to be updated for changing number of electrons 
                    recombination = np.random.exponential(lifetime)
                else:
                    recombination = np.array([])     
                i=0
                t0 = 0 #check whether t0 is correct (how it is reset)
                exp_duration = dt_filling*mc.steps*0.6 #to ensure that array is big enough
                print_param = 100
                fill_counter = 0
                while timebin[i-1]<mc.duration:
                    #timenow = timebin[i-1]
                    #Time step is decided by time until next recombination or filling event
                    dt_recomb = np.min(recombination - e_timer[:recombination.shape[0]]) if recombination.size > 0 else dt_filling-t0 #check this timebin subtraction
                    dt = np.min((dt_recomb,dt_filling-t0))#2C max step
                    if dt == dt_filling-t0:
                        t0 = 0
                        timebin[i] = timebin[i - 1] + dt
                        #if np.random.rand(1)+1 > ratio_traps:
                        fill_counter+=1
                        electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                        e_timer = np.append(e_timer,timebin[i])
                        distances = recalc_distances(electrons, holes,distances)
                        min_distances, hole_index = min_distance(distances)
                        lifetime = lifetime_tunneling(run_cfg, min_distances)
                        recombination = np.random.exponential(lifetime)
                        dt_filling = filling_time(run_cfg,electrons.shape[0],e_max,D)
                        #else:
                        #    pass
                    elif dt == dt_recomb:
                        dt = np.max((dt,0))
                        timebin[i] = timebin[i - 1] + dt

                        #Update time since last filling as well as filling time
                        t0 += dt
                        dt_filling = filling_time(run_cfg,electrons.shape[0],e_max,D)
                        #Handle recombination event
                        distances, electrons, holes,hole_index,min_distances,recombination,Lum[i,j,key],e_timer = recomber(run_cfg,
                            recombination,electrons,holes,hole_index,distances,min_distances,timebin[i],e_timer)
                        
                        ###UPDATE THE FILLING when electrons are removed 

                        #Recalculate distances for electrons that shared the recombined hole
                        lifetime = lifetime_tunneling(run_cfg, min_distances)
                        recombination = np.random.exponential(lifetime)
                    electron_ratio[i,j,key] = electrons.shape[0]/e_max

                    if i%print_param  == 0:
                        # Pause briefly (if needed), then move the cursor up two lines.
                        # \033[F moves the cursor up one line.
                        if i!= 0:
                            print("\033[F\033[F\033[F\033[F\033[F\033[F", end="")  # Moves the cursor up two lines

                        # Print the two lines initially.
                        print(f"Step {i+1} of {mc.steps} | Sim {j+1} of {mc.sims} | Experiment {key} of {exp_list}")
                        print(f"Trap filling ratio is {electron_ratio[i,j,key]:.2f}", flush=True)
                        print(f"Lum in last {print_param} steps is {np.sum(Lum[i-print_param:i,j,key])}", flush=True)
                        print(f"{timebin[i]/(3600*24*365*1000):.2f} thousand years passed of max {exp_duration/(3600*24*365*1000):.2f}", flush=True)
                        print(f"Time passed in last {print_param} steps: {(timebin[i] - timebin[i-print_param])/(3600*24*365*1000):.2f} thousand years", flush=True)
                        print(f"Filling events in last {print_param} steps: {fill_counter}", flush=True)
                        fill_counter= 0
                    if electron_ratio[i,j,key] > 0.98:
                        break
                    i+=1
                print(f"Simulation {j} done and we simulated across {timebin[i-1]/(3600*24*365*1000)} thousand years")

                x_ax[:,j,key] = timebin
            return x_ax, Lum, electron_ratio

def sim_lab_TL(run_cfg, Lum, x_ax, electron_ratio,key,exp_list):
            mc = run_cfg.exp_type_fp
            cfg_experiments  = pd.read_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IRSL50_0.25KperGy.csv")

            for j in range(mc.sims):
                electrons, holes,box_dim,e_max = initialize_box_bg(run_cfg)
                e_timer = np.array([])
                dt_filling = filling_time(run_cfg,electrons.shape[0],e_max)
                timebin = np.zeros(mc.steps)           
                distances = calc_distances(electrons, holes)
                if distances.size != 0:
                    min_distances, hole_index = min_distance(distances)
                
                    #find lifetime of all electrons
                    lifetime = lifetime_tunneling(run_cfg, min_distances) #this needs to be updated for changing number of electrons 
                    recombination = np.random.exponential(lifetime)
                else:
                    recombination = np.array([])     
                i=0
                t0 = 0 #check whether t0 is correct (how it is reset)
                exp_duration = dt_filling*mc.steps*0.6 #to ensure that array is big enough
                print_param = 50
                fill_counter = 0
                T = mc.T_start+273.15
                T_rate = (mc.T_end-mc.T_start)/mc.duration
                while timebin[i-1]<mc.duration:
                    #timenow = timebin[i-1]
                    #Time step is decided by time until next recombination or filling event
                    dt_recomb = np.min(recombination - e_timer[:recombination.shape[0]]) if recombination.size > 0 else dt_filling-t0 #check this timebin subtraction
                    dt = np.min((dt_recomb,dt_filling-t0))#2C max step
                    T = T + dt*T_rate
                    if dt == dt_filling-t0:
                        t0 = 0
                        timebin[i] = timebin[i - 1] + dt
                        #if np.random.rand(1)+1 > ratio_traps:
                        fill_counter+=1
                        electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                        e_timer = np.append(e_timer,timebin[i])
                        distances = recalc_distances(electrons, holes,distances)
                        min_distances, hole_index = min_distance(distances)
                        lifetime = lifetime_thermal(run_cfg, min_distances,T)
                        recombination = np.random.exponential(lifetime)
                        dt_filling = filling_time(run_cfg,electrons.shape[0],e_max)
                        #else:
                        #    pass
                    elif dt == dt_recomb:
                        dt = np.max((dt,0))
                        timebin[i] = timebin[i - 1] + dt

                        #Update time since last filling as well as filling time
                        t0 += dt
                        dt_filling = filling_time(run_cfg,electrons.shape[0],e_max)
                        #Handle recombination event
                        distances, electrons, holes,hole_index,min_distances,recombination,Lum[i,j,key],e_timer = recomber(run_cfg,
                            recombination,electrons,holes,hole_index,distances,min_distances,timebin[i],e_timer)
                        
                        ###UPDATE THE FILLING when electrons are removed 

                        #Recalculate distances for electrons that shared the recombined hole
                        lifetime = lifetime_thermal(run_cfg, min_distances,T)
                        recombination = np.random.exponential(lifetime)
                    electron_ratio[i,j,key] = electrons.shape[0]/e_max

                    if i%print_param  == 0:
                        # Pause briefly (if needed), then move the cursor up two lines.
                        # \033[F moves the cursor up one line.
                        if i!= 0:
                            print("\033[F\033[F\033[F\033[F\033[F\033[F", end="")  # Moves the cursor up two lines
                        else:
                            print("\033[F", end="")
                        # Print the two lines initially.
                        print(f"Step {i+1} of {mc.steps} | Sim {j+1} of {mc.sims} | Experiment {key} of {exp_list}")
                        print(f"Trap filling ratio is {electron_ratio[i,j,key]:.2f}", flush=True)
                        print(f"Lum in last {print_param} steps is {np.sum(Lum[i-print_param:i,j,key])}", flush=True)
                        print(f"{timebin[i]:.2f} seconds passed of max {exp_duration:.2f}", flush=True)
                        print(f"Time passed in last {print_param} steps: {(timebin[i] - timebin[i-print_param]):.2f} seconds", flush=True)
                        print(f"Filling events in last {print_param} steps: {fill_counter}", flush=True)
                        fill_counter= 0
                    if electron_ratio[i,j,key] > 0.98:
                        break
                    i+=1
                print(f"Simulation {j} done and we simulated across {timebin[i-1]} thousand years")

                x_ax[:,j,key] = timebin
            return x_ax, Lum, electron_ratio


def sim_lab_TL_residuals(run_cfg,params):
            rho_prime, E, s, N_e = params
            mc = run_cfg.exp_type_fp
            lab_cfg  = pd.read_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IRSL50_0.25KperGy.csv")
            SE = []
            print(f"rho': {mc.rho_prime}")
            for k in range(len(lab_cfg)):
                for j in range(mc.sims):
                    electrons, holes,box_dim,e_max = initialize_box_bg(run_cfg)
                    e_timer = np.array([])
                    dt_filling = filling_time(run_cfg,electrons.shape[0],e_max)
                    timebin = np.zeros(mc.steps)           
                    distances = calc_distances(electrons, holes)
                    if distances.size != 0:
                        min_distances, hole_index = min_distance(distances)
                    
                        #find lifetime of all electrons
                        lifetime = lifetime_tunneling(run_cfg, min_distances) #this needs to be updated for changing number of electrons 
                        recombination = np.random.exponential(lifetime)
                    else:
                        recombination = np.array([])     
                    i=0
                    t0 = 0 #check whether t0 is correct (how it is reset)
                    T = lab_cfg.T_start[k]+273.15
                    T_rate = (lab_cfg.T_end[k]-lab_cfg.T_start[k])/lab_cfg.Duration[k]
                    while timebin[i-1]<lab_cfg.Duration[k]:
                        #timenow = timebin[i-1]
                        #Time step is decided by time until next recombination or filling event
                        dt_recomb = np.min(recombination - e_timer[:recombination.shape[0]]) if recombination.size > 0 else dt_filling-t0 #check this timebin subtraction
                        dt = np.min((dt_recomb,dt_filling-t0))#2C max step
                        T = T + dt*T_rate
                        if dt == dt_filling-t0:
                            t0 = 0
                            timebin[i] = timebin[i - 1] + dt
                            #if np.random.rand(1)+1 > ratio_traps:
                            electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                            e_timer = np.append(e_timer,timebin[i])
                            distances = recalc_distances(electrons, holes,distances)
                            min_distances, hole_index = min_distance(distances)
                            lifetime = lifetime_thermal(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)
                            dt_filling = filling_time(run_cfg,electrons.shape[0],e_max)
                            #else:
                            #    pass
                        elif dt == dt_recomb:
                            dt = np.max((dt,0))
                            timebin[i] = timebin[i - 1] + dt


                            #Handle recombination event
                            distances, electrons, holes,hole_index,min_distances,recombination,_,e_timer = recomber(run_cfg,
                                recombination,electrons,holes,hole_index,distances,min_distances,timebin[i],e_timer)
                            
                            ###UPDATE THE FILLING when electrons are removed 
                            #Update time since last filling as well as filling time
                            t0 += dt
                            dt_filling = filling_time(run_cfg,electrons.shape[0],e_max)
                            #Recalculate distances for electrons that shared the recombined hole
                            lifetime = lifetime_thermal(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)

                        i+=1
                    #Calculate MSE
                    e_ratio_end = electrons.shape[0]/e_max
                    print(e_ratio_end-lab_cfg.Fill[k])
                    
                    SE.append((e_ratio_end-lab_cfg.Fill[k])**2)
            MSE = np.mean(SE)
            print(f"Mean Squared Error: {MSE} with params {params}")
            return MSE

def sim_lab_TL_residuals_iso(run_cfg):
            mc = run_cfg.exp_type_fp
            phys = run_cfg.physics_fp
            lab_cfg  = pd.read_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IR50_ISO.csv")
            SE = []
            ER = []
            print(f"rho': {mc.rho_prime}")
            for k in range(0,lab_cfg.exp_no.iloc[-1]):
                e_ratio_start = lab_cfg[lab_cfg["exp_no"]==k]["e_ratio"].iloc[0]

                for j in range(mc.sims):
                    electrons, holes,box_dim = initialize_box_bg(run_cfg,e_ratio_start)
                    e_timer = np.zeros(electrons.shape[0])
                    D = lab_cfg[lab_cfg["exp_no"]==k]["dose"].iloc[0]
                    dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                    timebin = np.zeros(mc.steps)           
                    distances = calc_distances(electrons, holes)
                    if distances.size != 0:
                        min_distances, hole_index = min_distance(distances)
                    
                        #find lifetime of all electrons
                        lifetime = lifetime_tunneling(run_cfg, min_distances) #this needs to be updated for changing number of electrons 
                        recombination = np.random.exponential(lifetime)
                    else:
                        recombination = np.array([])     
                    i=0
                    t0 = 0 #check whether t0 is correct (how it is reset)
                    T = lab_cfg[lab_cfg["exp_no"]==k]["temp"].iloc[0]+273.15
                    duration = np.max(lab_cfg[lab_cfg["exp_no"]==k]["time"])
                    obs_times = lab_cfg[lab_cfg["exp_no"]==k]["time"].to_numpy() #this is all the times at which we have data to compare with
                    e_ratio = np.zeros(obs_times.shape[0]) #bucket to calc e_ration at specific times
                    while timebin[i-1] < duration:
                        #timenow = timebin[i-1]
                        #Time step is decided by time until next recombination or filling event
                        dt_recomb = np.min((recombination + e_timer[:recombination.shape[0]])-timebin[i-1]) if recombination.size > 0 else dt_filling-t0 #check this timebin subtraction
                        dt = np.min((dt_recomb,dt_filling-t0))#2C max step
                        if dt == dt_filling-t0:
                            t0 = 0
                            timebin[i] = timebin[i - 1] + dt
                            #if np.random.rand(1)+1 > ratio_traps:
                            electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                            e_timer = np.append(e_timer,timebin[i])
                            distances = recalc_distances(electrons, holes,distances)
                            min_distances, hole_index = min_distance(distances)
                            lifetime = lifetime_thermal(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)
                            dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                            #else:
                            #    pass
                        elif dt == dt_recomb:
                            dt = np.max((dt,0))
                            timebin[i] = timebin[i - 1] + dt
                            if (np.where((timebin[i] >= recombination+e_timer))[0]).shape[0] > 1 and dt != 0:
                                print("More than one recombination event at the same time")

                            #Handle recombination event
                            distances, electrons, holes,hole_index,min_distances,recombination,_,e_timer = recomber(run_cfg,
                                recombination,electrons,holes,hole_index,distances,min_distances,timebin[i],e_timer)
                            
                            ###UPDATE THE FILLING when electrons are removed 
                            #Update time since last filling as well as filling time
                            t0 += dt
                            dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                            #Recalculate distances for electrons that shared the recombined hole
                            lifetime = lifetime_thermal(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)
                        while timebin[i]>obs_times[0]:
                            #Calculate MSE
                            e_ratio = electrons.shape[0]/mc.N_e
                            e_ratio_observed = (lab_cfg[(lab_cfg["exp_no"]==k) & (lab_cfg["time"]==obs_times[0])]["e_ratio"].iloc[0])
                            SE.append((e_ratio-e_ratio_observed)**2)
                            ER.append(abs(e_ratio-e_ratio_observed))
                            obs_times = np.delete(obs_times,0)
                            if len(obs_times) == 0:
                                break
                        i+=1
                    if len(obs_times)!= 0:
                        print(f"Sim ran for {timebin[i-1]} but real experiment ran for {obs_times[-1]} with {electrons.shape[0]} electrons, loop  {i}")
                        
    
                    #Calculate MSE
                    #e_ratio_end = electrons.shape[0]/mc.N_e
                    #e_ratio_observed = lab_cfg[lab_cfg["exp_no"]==k]["L"]
                    #print(e_ratio_end-lab_cfg.Fill[k])
                    
                    #SE.append((e_ratio_end-lab_cfg.L[k])**2)
            MSE = np.mean(SE)
            avgER = np.mean(ER)
            if MSE < 0.01:
                print("what")
            print(f"absError: {avgER}, MSE: {MSE} with params: \n rho' = {run_cfg.exp_type_fp.rho_prime}, E = {run_cfg.physics_fp.E}, b = {run_cfg.physics_fp.b}, alpha = {phys.alpha},holes= {run_cfg.exp_type_fp.holes}")
            return MSE