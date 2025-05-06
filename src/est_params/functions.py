from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
import numpy as np
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT
import pandas as pd
from matplotlib.lines import Line2D



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
        # For each category (exp_type_fp, physics_fp) create a sub-dictionary.
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
            run_dict[sub_name] = OmegaConf.create(sub_run)
        runs[i] = OmegaConf.create(run_dict)
    return runs

def initialize_box_bg(cfg,e_ratio_start=0):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    phys = cfg.physics_fp
    mc = cfg.exp_type_fp
    e = int(mc.N_e*e_ratio_start)

    rho = rho_func(cfg)
    h = int(mc.holes)
    d = (h/rho)**(1/3)
    box_l, box_w, box_h = d, d, d
    box_volume = box_l*box_w*box_h
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

def rho_func(cfg):
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    rho = mc.rho_prime*(3/(4*np.pi)*phys.alpha**3)
    return rho

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


def lifetime_tunneling(cfg,distances):
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    lifetime = 1/phys.b*np.exp(phys.alpha*distances)
    return lifetime


def lifetime_fading(cfg,distances, T=0):
    """ Compute the combined thermal (conduction-band) and localized (tunnelling)
    recombination lifetime for each electron-hole pair.
    temp should be in kelvin."""
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    # Conduction-band release rate: s * exp(-E_cb / (phys.k_b * T))
    k_cb = phys.s * np.exp(-phys.E_cb / (phys.k_b * T))

    # Localized tunnelling rate: b * exp(-E_loc / (phys.k_b * T) - phys.alpha * distances)
    if np.random.rand(1) > phys.Retrap:
        k_tun = phys.b * np.exp(-phys.E_loc_1 / (phys.k_b * T) - phys.alpha * distances)
    else: 
        k_tun = phys.b * np.exp(-phys.E_loc_2 / (phys.k_b * T) - phys.alpha * distances)

    # Total recombination rate is sum of the two channels
    rate = k_cb + k_tun

    # Lifetime is inverse of total rate
    lifetime = 1.0 / rate

    return lifetime

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
    return np.random.exponential(lifetime)

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
        hole_index[hole_index > hole_index[recombination_idx[0]]] -= 1 
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
    holes_that_recombined = hole_index[recombination_idx]  

    # Boolean mask: does hole_index[i] match any of the holes_that_recombined?
    mask_hole_match = np.isin(hole_index, holes_that_recombined)

    # Boolean mask: is electron index i in recombination_idx?
    mask_not_recombining_e = ~np.isin(np.arange(len(hole_index)), recombination_idx)

    # Combine the two conditions:
    electrons_new_d = np.where(mask_hole_match & mask_not_recombining_e)[0]
    return electrons_new_d

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
    new_hole = (np.random.rand(1, 3)) * scaling_b #shoud it instead be one hole added in the box and then maybe one in boundary
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

def recomber(cfg,recombination,electrons,holes,hole_index,distances,min_distances,time,e_timer=np.zeros([1,1])):
    """
    Recalculate the recombination time and remove the electron and hole that recombined
    """
    #Check what electrons have exceeded their lifetimes taking into account when they got filled
    recombination_idx = np.where((time >= recombination+e_timer))[0] 
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
    return distances, electrons, holes,hole_index,min_distances,recombination,Lum,e_timer


def analy_TL_iso(cfg, timebin,i):
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    ns = mc.electrons*np.exp(-mc.rho_prime*np.log(1.8*phys.s*timebin)**3)   
    return ns                 

def sim_lab_TL_residuals(run_cfg, lab_data:str="CLBR_IRSL50_0.25KperGy",PLOT = False):
            mc = run_cfg.exp_type_fp
            phys = run_cfg.physics_fp
            lab_cfg  = pd.read_csv(f"{PROJECT_ROOT}/data/processed/{lab_data}.csv")
            SE = []
            ER = []
            TSS = []
            # Dictionary to collect time and electron_ratio series for each outer loop (lab_cfg row)
            plot_data = {}
            plot_data_full = {}
            for k in range(len(lab_cfg)):
                e_ratio_start = lab_cfg["e_ratio"].iloc[k]
                # Create local lists to record time and electron ratio for this simulation run.
                sim_times = []
                sim_ratios = []
                
                # Initialize dictionary entry for this outer loop
                plot_data[k] = {"times": [], "ratios": []}
                plot_data_full[k] = {"times": [], "ratios": []}
                for j in range(mc.sims):
                    electrons, holes,box_dim = initialize_box_bg(run_cfg, e_ratio_start)
                    D = phys.D
                    dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                    timebin = np.zeros(mc.steps)       
                    T = lab_cfg.T_start[k]+273.15    
                    distances = calc_distances(electrons, holes)
                    if distances.size != 0:
                        min_distances, hole_index = min_distance(distances)
                        #find lifetime of all electrons
                        lifetime = lifetime_fading(run_cfg, min_distances,T) 
                        recombination = np.random.exponential(lifetime)
                    else:
                        recombination = np.array([])     
                    i=0
                    T_rate = (lab_cfg.T_end[k]-lab_cfg.T_start[k])/lab_cfg.Duration[k]
                    while timebin[i-1]<lab_cfg.Duration[k]:
                        #Time step is decided by time until next recombination or filling event
                        dt_recomb = np.min(recombination) if recombination.size > 0 else dt_filling 
                        dt = np.min((dt_recomb,dt_filling))

                        if dt == dt_filling:
                            dt = max((dt,0))
                            T = T + dt*T_rate
                            timebin[i] = timebin[i - 1] + dt

                            electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                            distances = recalc_distances(electrons, holes,distances)
                            min_distances, hole_index = min_distance(distances)
                            lifetime = lifetime_fading(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)
                            dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                        else:
                            dt = np.max((dt,0))
                            T = T + dt*T_rate
                            timebin[i] = timebin[i - 1] + dt
                            #Handle recombination event
                            distances, electrons, holes,hole_index,min_distances,_,_,_ = recomber(run_cfg,
                                recombination,electrons,holes,hole_index,distances,min_distances,timebin[i])
                            if np.random.rand(1) < phys.Retrap:
                                electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                                distances = recalc_distances(electrons, holes,distances)
                                min_distances, hole_index = min_distance(distances)
                            #Recalculate waiting times:
                            lifetime = lifetime_fading(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)
                            dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                                                # Record the current electron ratio and simulation time.
                        e_ratio_current = electrons.shape[0] / mc.N_e

                        i+=1
                        sim_times.append(timebin[i-1])
                        sim_ratios.append(e_ratio_current)
                    if j == mc.sims-1:
                        plot_data[k]["times"] = lab_cfg.Duration[k]
                        plot_data[k]["ratios"] = np.mean(sim_ratios[-1])
                        plot_data_full[k]["times"] = sim_times
                        plot_data_full[k]["ratios"] = sim_ratios
                    #Calculate MSE
                    e_ratio_end = electrons.shape[0]/mc.N_e
                    ER.append(abs(e_ratio_end-lab_cfg.Fill[k]))
                    SE.append((e_ratio_end-lab_cfg.Fill[k])**2)
                    TSS.append((lab_cfg.Fill[k]-np.mean(lab_cfg.Fill))**2)
            MSE = np.mean(SE)
            R2 = 1 - (np.sum(SE) / np.sum(TSS))
            avgER = np.mean(ER)
            if PLOT:
                plot_e_ratio_timeseries(lab_cfg, plot_data,lab_data)
                plot_e_ratio_timeseries(lab_cfg, plot_data_full,lab_data,plot_type = "full")
            printer(run_cfg, avgER, MSE)
            return MSE

def sim_lab_TL_residuals_iso(run_cfg, lab_data:str = "CLBR_IR50_ISO", PLOT:bool = False):
            mc = run_cfg.exp_type_fp
            phys = run_cfg.physics_fp
            lab_cfg  = pd.read_csv(f"{PROJECT_ROOT}/data/processed/{lab_data}.csv")
            SE = []
            ER = []
            TSS = []
            # Dictionary to collect time and electron_ratio series for each outer loop (lab_cfg row)
            plot_data = {}
            for k in range(0,lab_cfg.exp_no.iloc[-1]):
                e_ratio_start = lab_cfg[lab_cfg["exp_no"]==k]["e_ratio"].iloc[0]
                # Create local lists to record time and electron ratio for this simulation run.
                sim_times = []
                sim_ratios = []
                
                #Initialize dictionary entry for this outer loop
                plot_data[k] = {"times": [], "ratios": []}
                for j in range(mc.sims):
                    electrons, holes,box_dim = initialize_box_bg(run_cfg,e_ratio_start)
                    D = lab_cfg[lab_cfg["exp_no"]==k]["dose"].iloc[0]
                    dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                    timebin = np.zeros(mc.steps)           
                    distances = calc_distances(electrons, holes)
                    T = lab_cfg[lab_cfg["exp_no"]==k]["temp"].iloc[0]+273.15
                    if distances.size != 0:
                        min_distances, hole_index = min_distance(distances)
                    
                        #find lifetime of all electrons
                        lifetime = lifetime_fading(run_cfg, min_distances,T)
                        recombination = np.random.exponential(lifetime)
                    else:
                        recombination = np.array([])     
                    i=0
                    duration = np.max(lab_cfg[lab_cfg["exp_no"]==k]["time"])
                    obs_times = lab_cfg[lab_cfg["exp_no"]==k]["time"].to_numpy() #this is all the times at which we have data to compare with
                    e_ratio = np.zeros(obs_times.shape[0]) #bucket to calc e_ratio at specific times
                    while timebin[i-1] < duration:
                        #Time step is decided by time until next recombination or filling event
                        dt_recomb = np.min(recombination) if recombination.size > 0 else dt_filling
                        dt = np.min((dt_recomb,dt_filling))
                        if dt == dt_filling:
                            timebin[i] = timebin[i - 1] + dt
                            electrons, holes = add_electron(run_cfg,box_dim,electrons,holes)
                            dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                            distances = recalc_distances(electrons, holes,distances)
                            min_distances, hole_index = min_distance(distances)
                            lifetime = lifetime_fading(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)

                        elif dt == dt_recomb:
                            dt = np.max((dt,0))
                            timebin[i] = timebin[i - 1] + dt

                            #Handle recombination event
                            distances, electrons, holes,hole_index,min_distances,_,_,_ = recomber(run_cfg,
                                recombination,electrons,holes,hole_index,distances,min_distances,timebin[i])
                            
                            dt_filling = filling_time(run_cfg,electrons.shape[0],mc.N_e,D)
                            lifetime = lifetime_fading(run_cfg, min_distances,T)
                            recombination = np.random.exponential(lifetime)
                        while timebin[i]>obs_times[0]:
                            #Calculate MSE
                            e_ratio = electrons.shape[0]/mc.N_e
                            e_ratio_observed = (lab_cfg[(lab_cfg["exp_no"]==k) & (lab_cfg["time"]==obs_times[0])]["e_ratio"].iloc[0])
                            SE.append((e_ratio-e_ratio_observed)**2)
                            TSS.append((e_ratio_observed-np.mean((lab_cfg[(lab_cfg["exp_no"]==k) & (lab_cfg["time"]==obs_times[0])]["e_ratio"]))**2))
                            ER.append(abs(e_ratio-e_ratio_observed))
                            obs_times = np.delete(obs_times,0)
                            sim_times.append(timebin[i])
                            sim_ratios.append(e_ratio)
                            if len(obs_times) == 0:
                                break
                        i+=1
                    plot_data[k]["times"] = sim_times
                    plot_data[k]["ratios"] = sim_ratios
                    if len(obs_times)!= 0:
                        print(f"Sim ran for {timebin[i-1]} but real experiment ran for {obs_times[-1]} with {electrons.shape[0]} electrons, loop  {i}")
            MSE = np.mean(SE)
            R2 = 1 - (np.sum(SE)/np.sum(TSS))
            avgER = np.mean(ER)
            printer(run_cfg, avgER, MSE)
            if PLOT:
                rows = []
                for sim_id, data in plot_data.items():
                    for t, r in zip(data["times"], data["ratios"]):
                        rows.append({
                            "simulation": sim_id,
                            "time": t,
                            "ratio": r
                        })

                df = pd.DataFrame(rows)
                # write to CSV
                df.to_csv(f"{PROJECT_ROOT}/results/lab_sims/iso_data.csv", index=False)
                plot_e_ratio_timeseries_iso(lab_cfg, df,lab_data)
            return MSE

def printer(run_cfg, avgER, MSE):
    print(f"""absError: {avgER}, MSE: {MSE} with params: 
        rho_prime = {run_cfg.exp_type_fp.rho_prime}, E_cb = {run_cfg.physics_fp.E_cb}, D0 = {run_cfg.physics_fp.D0}, 
        E_loc_1 = {run_cfg.physics_fp.E_loc_1},E_loc_2 = {run_cfg.physics_fp.E_loc_2}, s = {run_cfg.physics_fp.s}, b = {run_cfg.physics_fp.b}, 
        alpha = {run_cfg.physics_fp.alpha},holes= {run_cfg.exp_type_fp.holes}, P_retrap = {run_cfg.physics_fp.Retrap},""")
    print('\033[5A')
            


def plot_e_ratio_timeseries(lab_cfg, plot_data,exp_type,plot_type = "slim"):
    L0 = 1.52 # Luminescence filling constant
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Gather all T_end values used in lab_cfg (one per row)
    #    For each k in plot_data, we read off the T_end from lab_cfg.
    try:
        T_ends = [lab_cfg["T_end"][k] for k in plot_data.keys()]
    except KeyError as e:
        try:
            lab_cfg["T_end"] = lab_cfg["temp"]
            T_ends = [lab_cfg["T_end"].iloc[k] for k in plot_data.keys()]
        except KeyError as e:
            raise KeyError("lab_cfg must contain a column 'T_end' or 'temp'") from e

    try:
        lab_cfg["Duration"] = lab_cfg["time"]
    except KeyError as e:
        print("Attempting to plot")
    #Create a sorted list of unique T_end values
    unique_ends = sorted(set(T_ends))
    cmap = plt.cm.get_cmap('viridis', len(unique_ends))

    #Create a dictionary: T_end -> color
    color_map = {val: cmap(i) for i, val in enumerate(unique_ends)}

    #Sort plot_data items by T_end (so lines plot in ascending T_end order)
    #    Note: x[0] is the key 'k', so we find its T_end in lab_cfg.
    sorted_plot_data = sorted(plot_data.items(), key=lambda x: lab_cfg["T_end"][x[0]])

    handles_dict = {}

    #Plot each time series using the color associated with its T_end
    CUTOFF = 17000 #just bcs we have this 1e20 duration that sometimes get used to stop sims
    for k, data in sorted_plot_data:
        times = np.array(data["times"])
        ratios = np.array(data["ratios"])
        # build a mask of all points ≤ cutoff
        mask = times <= CUTOFF
            # if nothing left, skip this curve entirely
        if not mask.any():
            continue
        # apply the mask
        times_filt  = times[mask]
        ratios_filt = ratios[mask]
        t_end = lab_cfg["T_end"][k]
        color = color_map[t_end]

        # plot the filtered curve
        line = ax.plot(times_filt, ratios_filt, color=color,
                    label=f"T_end = {t_end}C")

        # mark the last (filtered) point  
        ax.plot(times_filt[-1], ratios_filt[-1],
                marker='o', markersize=8, markeredgecolor='black',
                markerfacecolor=color, linestyle='')
            
        # If we haven't seen this t_end before, store the handle for the legend
        if t_end not in handles_dict:
            handles_dict[t_end] = line

    # 7. Build a custom legend so each T_end is shown only once
    handles = []
    labels = []
    for val in unique_ends:
        if val in handles_dict:
            handles.append(handles_dict[val][0])
            labels.append(f"{val}C")

    # 8. Plot observed laboratory values on the same axes
    try:
        lab_cfg["Fill"] = lab_cfg["L"]/L0
    except KeyError as e:
        print("Attempting to plot")
    obs_handle = ax.scatter(
        lab_cfg["Duration"], lab_cfg["Fill"],
        color="red", marker="x", label="Observed",
        zorder=10
    )
    handles.append(obs_handle)
    labels.append("Observed")

    ax.legend(handles, labels, title="T_end:")

    # 8. Final labeling
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ratio of filled electron traps")
    ax.set_ylim(0,1)
    ax.set_title(f"Evolution of Electron Ratio Over Time\n {exp_type} ")
    plt.savefig(f"{PROJECT_ROOT}/results/plots/lab/{exp_type}_{plot_type}.png")


def plot_e_ratio_timeseries_iso(lab_cfg, df,exp_type,plot_type = "slim"):
    L0 = 1.52 # Luminescence filling constant
    fig, ax = plt.subplots(figsize=(10, 6))

    n = lab_cfg["exp_no"].nunique()
    cmap = plt.get_cmap("tab10")          
    col = cmap(np.arange(n)) 
    for i in range(lab_cfg["exp_no"].iloc[-1]):
        sim_data = df[df["simulation"]==i]
        lab_data = lab_cfg[lab_cfg["exp_no"]==i]/L0
        plt.plot(lab_data["time"], sim_data["ratio"], linestyle="None", marker="o", label=f"Temp {lab_data['temp'].iloc[0]}",color=col[i])
        ax.scatter(
        lab_data["time"], lab_data["L"],
        color=col[i], marker="x", label="Observed",
        zorder=10
        )  
        
    plt.xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ratio of filled electron traps")
    ax.set_ylim(0,1)
    ax.set_title(f"Evolution of Electron Ratio Over Time\n {exp_type} ")
    
    # 1) a little legend box for the temperatures
    temp_handles = []
    # pull out the unique experiment numbers & temps in the same order they are plotted
    exp_nos = lab_cfg["exp_no"].unique()
    temps   = [lab_cfg[lab_cfg["exp_no"]==i]["temp"].iloc[0] for i in exp_nos]

    for i, T in enumerate(temps):
        temp_handles.append(
            Line2D([0],[0],
                linestyle="None", marker="o", 
                color=col[i], markersize=8,
                label=f"Temperature: {T} °C")
        )
    # place  legend in the upper right
    temp_legend = ax.legend(handles=temp_handles,
                            title="Experiment Temperatures",
                            loc="center left",
                            frameon=True)
    ax.add_artist(temp_legend)


    # 2) a second legend box for “simulated vs lab”
    data_handles = [
        Line2D([0],[0],
            linestyle="None", marker="o",
            color="gray", markersize=8,
            label="Simulated data"),
        Line2D([0],[0],
            linestyle="None", marker="x",
            color="black", markersize=8,
            label="Lab data")
    ]
    ax.legend(handles=data_handles,
            loc="lower right",
            frameon=True)

    plt.savefig(f"{PROJECT_ROOT}/results/plots/lab/{exp_type}_{plot_type}.png")
