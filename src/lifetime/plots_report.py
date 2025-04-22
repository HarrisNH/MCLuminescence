from omegaconf import DictConfig, OmegaConf
from paths import CONFIG_DIR, PROJECT_ROOT
from omegaconf.listconfig import ListConfig
import numpy as np
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT
import pandas as pd
import hydra
import os
from matplotlib import cm

BOUNDS_FADING = [
    (1e-8, 1e-3),  # rho_prime
    (0.8,   2.2),  # E_cb
    (0.5,   1.8),  # E_loc
    (1e12, 1e14),  # s
    (1e10, 1e13),  # b
    (1e9,  5e10),  # alpha
    (1e2,    750)  # holes
]

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

def initialize_box_bg(cfg,e_ratio_start=0, rho: int=1e16):
    """
    Initialize the electrons and holes in the box centered at the origin.
    """
    phys = cfg.physics_fp
    mc = cfg.exp_type_fp
    e = int(mc.N_e*e_ratio_start)

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


def theoretical_nearest_neighbor_dist(cfg, r,rho):
    mc = cfg.exp_type_fp
    NND = np.exp(-4/3 * np.pi * rho * r**3) * 4 * np.pi * rho * r**2  # densitydistribution of holes
    return NND

def distance_plot(cfg, distances,rho):
    # Convert distances from metres to nanometres for plotting
    distances_nm = distances * 1e9
    # Create an array of r values in metres and then convert to nm for x-axis plotting
    r_m = np.linspace(0, np.max(distances), 100)
    r_nm = r_m * 1e9
    # Calculate the theoretical nearest-neighbour distribution using r in metres
    dist_theoretical = theoretical_nearest_neighbor_dist(cfg, r_m,rho)/1e9 #down to nm
    if cfg.exp_type_fp.distance_plot:
        plt.hist(distances_nm, bins=30, density=True, alpha=0.7)
        plt.plot(r_nm, dist_theoretical, color="red", lw=2, label="Theoretical NND")
        plt.xscale("log")
        plt.xlabel("Distance (nm)")
        plt.ylabel("Probability density")
        plt.tight_layout()
        plt.show()
        print("plot")



def plot_all_distributions(cfg):

    # your array of number‐densities
    rhos = [1e14, 1e15, 1e16, 1e17, 1e18]
    cmap = cm.get_cmap('viridis', len(rhos))

    # prepare log‐spaced bins (in nm)
    bins_nm = np.logspace(0, 2.5, 200)   # from 1 nm up to ~300 nm

    # create a single figure/axes
    fig, ax = plt.subplots(figsize=(6,4))

    for idx, rho in enumerate(rhos):
        # re‐init your box
        cfg.exp_type_fp.holes = cfg.exp_type_fp.N_e = 1000
        electrons, holes, _ = initialize_box_bg(cfg, e_ratio_start=1, rho=rho)

        # compute all nearest‐neighbor distances (in metres)
        dists_m, _ = min_distance(electrons, holes)

        # convert to nm
        dists_nm = dists_m * 1e9

        # histogram → probability density
        pdf, edges = np.histogram(dists_nm, bins=bins_nm, density=True)
        centers = 0.5*(edges[:-1]+edges[1:])

        # plot as a “step” line, not filled bars
        ax.plot(centers, pdf,
                color=cmap(idx),
                lw=2,
                label=rf'$\rho=10^{{{int(np.log10(rho))}}}$')

    # Now plot the analytic nearest‐neighbor PDF once, on _top_.
    # We need the same rho0 in dimensionless form and α in 1/nm:
    alpha = cfg.physics_fp.alpha * 1e-9   # if α was in m⁻¹, now convert to nm⁻¹
    # pick any single rho, or loop if you want multiple.
    rho0 = rhos[-1]*(1e9**3)   # convert ρ [m⁻³] to [nm⁻³]
    r_nm = np.logspace(0, 2.5, 400)
    pdf_analytic = (
        4*np.pi * rho0 * r_nm**2
        * np.exp(-4*np.pi/3 * rho0 * r_nm**3)
    )
    ax.plot(r_nm, pdf_analytic, 'k--', lw=1.5, label='analytic')

    ax.set_xscale('log')
    ax.set_xlabel("Nearest‐neighbour distance $r$ (nm)")
    ax.set_ylabel("Probability density $p(r)$")
    ax.set_title("Distribution of $r$ for different number‐densities")
    ax.legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()



def analytical_BG_rad():
    D0 = 300
    t = np.logspace(0,13,1000)
    Dr = 1/(3155760000000) #Gy/s
    Dn = Dr*t
    rho_prime = 1e-6
    s = 3e15
    M = 587
    Ln = (1-np.exp(-Dn/D0))*M*np.exp(-rho_prime*np.log(((D0*s)/Dr)**3))
    plt.plot(t, Ln)
    plt.xscale('log')
    plt.xlim(1e0,1e13)
    print("Done")


def plot_parameter_distribution(exp_type: str="tl_fsm-13"):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, f"results/sims/result_{exp_type}.csv"))
    df.dropna(inplace=True)
    # 2) List your parameter columns
    params = ["param_0","param_1","param_2","param_3","param_4","param_5","param_6"]

    # 3) Compute sample mean and standard deviation
    means = df[params].mean()
    stds  = df[params].std()

    # 4) Plot a 3×3 grid of histograms
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()


    param_names = ['rho_prime', 'E_cb', 'E_loc', 's', 'b', 'alpha', 'holes']
    for idx, param in enumerate(params):
        ax = axes[idx]
        ax.set_xscale('log')
        data = df[param]
        # Histogram
        lo, hi = BOUNDS_FADING[idx]
        bins = np.logspace(np.log10(lo), np.log10(hi), 30)
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        
        # Mean line
        mu = means[param]
        sigma = stds[param]
        ax.axvline(mu, color='red', linestyle='--', linewidth=2, label=f"Mean = {mu:.3e}")
        # ±1σ shading
        ax.axvspan(mu - sigma, mu + sigma, color='red', alpha=0.2, label=f"±1σ ({sigma:.3e})")

        ax.set_title(param_names[idx])
        ax.legend()
        ax.set_xlim(BOUNDS_FADING[idx][0], BOUNDS_FADING[idx][1])
        
    
    # Remove extra subplot(s)
    for j in range(len(params), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Parameter distribution for {exp_type } experiment", fontsize=18)
    plt.tight_layout()

    plt.savefig(f"{PROJECT_ROOT}/results/plots/parameter_distribution_{exp_type}.png")

    plt.show()
    print("Parameters:")

def plot_parameter_distribution_mse(exp_type: str="tl_fsm-13"):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, f"results/sims/result_{exp_type}.csv"))
    df.dropna(inplace=True)
    params = ["param_0","param_1","param_2","param_3","param_4","param_5","param_6"]
    param_names = ['rho_prime', 'E_cb', 'E_loc', 's', 'b', 'alpha', 'holes']

    means = df[params].mean()
    stds  = df[params].std()

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        ax.set_xscale('log')

        data = df[param].values
        mse  = df['mse'].values    # ← make sure your DataFrame has an 'MSE' column

        # build log‐spaced bins between your desired bounds
        lo, hi = BOUNDS_FADING[idx]
        bins = np.logspace(np.log10(lo), np.log10(hi), 20)

        # 1) draw the histogram of counts
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlim(lo, hi)

        # 2) compute bin centres
        centers = (bins[:-1] + bins[1:]) / 2

        # 3) assign each point to a bin
        which_bin = np.digitize(data, bins) - 1  # now 0 … len(bins)-2

        # 4) compute mean MSE in each bin
        mean_mse_per_bin = []
        for bin_idx in range(len(bins)-1):
            sel = mse[which_bin == bin_idx]
            mean_mse_per_bin.append(sel.mean() if len(sel) else np.nan)
        mean_mse_per_bin = np.array(mean_mse_per_bin)

        # 5) plot mean‐MSE dots on a twin‐y axis
        ax2 = ax.twinx()
        ax2.scatter(centers, mean_mse_per_bin,
                    color='C1', marker='o', s=50, zorder=3,
                    label='mean MSE')
        ax2.set_ylabel("mean MSE", color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_yscale('linear')

        # 6) overlay mean ±1σ of the parameter on the histogram
        mu    = means[param]
        sigma = stds[param]
        ax.axvline(mu, color='red', linestyle='--', linewidth=2,
                   label=f"μ = {mu:.3e}")
        ax.axvspan(mu - sigma, mu + sigma, color='red', alpha=0.2,
                   label=f"±1σ = {sigma:.3e}")

        ax.set_title(param_names[idx])
        ax.legend(loc='upper left')

    # remove empty axes
    for j in range(len(params), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Parameter & mean‑MSE distribution for {exp_type}", fontsize=18)
    fig.tight_layout(rect=[0,0,1,0.96])

    plt.savefig(f"{PROJECT_ROOT}/results/plots/parameter_distribution_{exp_type}.png")
    plt.show()

plot_parameter_distribution_mse(exp_type="tl_clbr")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    rhos = [10**18, 10**15, 10**16, 10**17, 10**18]
    
    for i, rho in enumerate(rhos):
        # re‑init box
        cfg.exp_type_fp.distance_plot = True
        cfg.exp_type_fp.holes = 10000
        cfg.exp_type_fp.N_e = 10000
        electrons, holes, box_dim = initialize_box_bg(cfg, e_ratio_start=1, rho=rho)
        min_dists, _ = min_distance(electrons, holes)
        distance_plot(cfg, min_dists,rho)
        print(print(f"rho = {rho}"))

# def main(cfg: DictConfig) -> None:
#     rhos = [10**14, 10**15, 10**16, 10**17, 10**18]

#     # first pass: generate all the data so we know min/max
#     all_dists = []
#     datasets = []
#     for rho in rhos:
#         # re‑init box
#         cfg.exp_type_fp.holes = 1000
#         cfg.exp_type_fp.N_e = 1000
#         electrons, holes, box_dim = initialize_box_bg(cfg, e_ratio_start=1, rho=rho)
#         min_dists, _ = min_distance(electrons, holes)
#         d_nm = min_dists * 1e9
#         datasets.append((rho, d_nm))
#         all_dists.append(d_nm)

#     # flatten and find global range
#     all_dists = np.hstack(all_dists)
#     d_min, d_max = all_dists.min(), all_dists.max()

#     # build log‐spaced bins
#     bins = np.logspace(np.log10(d_min), np.log10(d_max), 100)

#     # now actually plot
#     plt.figure(figsize=(6,4))
#     cmap = plt.get_cmap('rainbow')
#     for i, (rho, d_nm) in enumerate(datasets):
#         plt.hist(d_nm,
#                  bins=bins,
#                  density=True,
#                  histtype='step',
#                  lw=1.5,
#                  color=cmap(i/len(rhos)),
#                  label=f"ρ = 1e{int(np.log10(rho))}")

#     plt.xscale('log')
#     plt.xlim(0,1e2)
#     plt.xlabel(r"Nearest‐neighbour distance $r$ (nm)")
#     plt.ylabel(r"Probability density $p(r)$")
#     plt.title("Distribution of $r$ for different number‐densities")
#     plt.legend(loc='upper right', fontsize='small')
#     plt.tight_layout()
#     plt.show()
    
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    # ----------------------------------------------------------------------------
    #  parameters for Eq. (10)
    # ----------------------------------------------------------------------------
    D0       = 300.0                   # Gy
    rho_prime= 1e-6                    # dimensionless density
    s        = 3e15                    # tunneling attempt-frequency [s^-1]
    M        = 587                     # total trap count (scaling)
    # natural dose rate: 1 Gy per ka
    DR       = 1.0 / (1e3 * 365.25*24*3600)  # Gy/s  ≈ 3.1688e-11

    # time axis (seconds), convert to cumulative dose Dn=DR*t
    t        = np.logspace(0, 13, 1000)     # seconds
    Dn       = DR * t                       # Gy

    # ----------------------------------------------------------------------------
    #  Eq. (10)  Ln(Dn) = [1 - exp(-Dn/D0)] * M * exp[-ρ' * (ln(D0*s/DR))^3]
    # ----------------------------------------------------------------------------
    fade_const = np.exp(-rho_prime * np.log(D0*s/DR)**3)
    Ln         = (1 - np.exp(-Dn/D0)) * M * fade_const

    # ----------------------------------------------------------------------------
    #  plot
    # ----------------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(t, Ln/587, '--', lw=2,color = "black")

    # 1) Use a linear x‐axis
    ax.set_xscale('linear')

    # 2) Define tick locations from 0 to 1e13 every 2e12
    ticks = np.arange(0, 1.0001e13, 2e12)

    # 3) Build matching labels
    labels = []
    for x in ticks:
        if x == 0:
            labels.append('0')
        elif x == 1e13:
            labels.append('1e+13')
        else:
            # e.g. 2e12, 4e12, 6e12, 8e12
            labels.append(f'{int(x/1e12)}e+12')

    # 4) Apply them
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    # 5) Polish
    ax.set_xlabel('Irradiation time, s')
    ax.set_ylabel('% filled traps')
    ax.set_xlim(0, 1e13)
    ax.set_ylim(0, 0.75)
    
    plt.savefig(f"{PROJECT_ROOT}/results/plots/analytical_BG_radiation_3.svg")

    main()
