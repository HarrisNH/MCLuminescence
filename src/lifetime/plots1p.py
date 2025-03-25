import os
from paths import PROJECT_ROOT,CONFIG_DIR
import hydra
from omegaconf import DictConfig, OmegaConf
import functions as fc
import numpy as np
import matplotlib.pyplot as plt
import sim1p as sim
import pandas as pd
from scipy.interpolate import make_interp_spline

def running_mean(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth `data` by computing the average of every `window_size` points.
    The result has fewer points if mode='valid'. 
    Use mode='same' if you want the same length as the original data.
    """
    kernel = np.ones(window_size) / window_size
    # Convolution-based smoothing
    smoothed = np.convolve(data, kernel, mode='valid')
    return smoothed

def TL_temp(cfg: DictConfig, x_ax, Lums) -> None:
    mc = cfg.exp_type_fp

    #output dirs
    output_dir_hydra = os.path.join(os.getcwd(), f"results/lifetime/TL/")
    output_dir_local = os.path.join(PROJECT_ROOT, f"results/lifetime/TL/")
    os.makedirs(output_dir_local, exist_ok=True)
    os.makedirs(output_dir_hydra, exist_ok=True)


    n_colors = Lums.shape[-1]
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    plt.clf()
    window_size = 50
    for i in range(Lums.shape[-1]):
        lum_across_sims = np.mean(Lums[:,:,i], axis=1) #mean across nsims for each temperature
        lum_across_sims_normalized = lum_across_sims / np.max(lum_across_sims)
        # Suppose lum_across_sims is your raw signal array
        smoothed_lum = running_mean(lum_across_sims, window_size=window_size)

        # If using 'valid' mode, you’ll need to adjust the x_ax to match lengths
        x_smoothed = x_ax[(window_size - 1) // 2 : -(window_size // 2)]

        plt.plot(x_smoothed, smoothed_lum[:x_smoothed.shape[0]],color=colors[i], linewidth=0.5, label=f'{mc.T_rate[i]}', linestyle='--')
    plt.text(0.6,0.9,"Heating Rate ($^{\circ}C.s^{-1}$)",
    transform=plt.gca().transAxes,   # interpret x,y as axes coords
    fontsize=12,)
    
    plt.xlabel("Temperature ($^{\circ}C$)")
    plt.ylabel("Intensity ($^{\circ}C^{-1}$)")
    plt.legend(bbox_to_anchor=(0.7, 0.9), loc='upper left')
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir_local, f"TL_dT{mc.T_rate}.svg"))
    plt.savefig(os.path.join(output_dir_hydra, f"TL_dT{mc.T_rate}.svg"))


def TL_lum_iso(cfg: DictConfig, x_ax, Lums) -> None:
    mc = cfg.exp_type_fp

    #output dirs
    output_dir_local = os.path.join(os.getcwd(), f"results/lifetime/isoTL/")
    output_dir_hydra = os.path.join(PROJECT_ROOT, f"results/lifetime/isoTL/")
    os.makedirs(output_dir_local, exist_ok=True)
    os.makedirs(output_dir_hydra, exist_ok=True)

    colors = plt.cm.rainbow(np.linspace(0, 1, Lums.shape[-1]))
    plt.clf()
    # Create two-subplot figure: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    for i in range(Lums.shape[-1]):
        lum_across_sims = np.mean(Lums[:,:,i], axis=1) #mean across nsims for each temperature
        lum_across_sims_normalized = lum_across_sims / np.max(lum_across_sims)

        # Plot unnormalized on the first subplot
        ax1.plot(
            x_ax, lum_across_sims[:x_ax.shape[0]],
            color=colors[i], linewidth=0.5,
            label=f"Temp: {mc.T_start[i]} $^\circ$C",
            linestyle="--"
        )

        # Plot normalized on the second subplot
        ax2.plot(
            x_ax, lum_across_sims_normalized[:x_ax.shape[0]],
            color=colors[i], linewidth=0.5,
            label=f"Temp: {mc.T_start[i]} $^\circ$C",
            linestyle="--"
        )

    # Unnormalized plot
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal")
    ax1.set_title("Unnormalized")

    # Normalized plot
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalized Signal")
    ax2.set_title("Normalized")

    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir_local, f"Time_Lum_C_{mc.T_start}.svg"))
    plt.savefig(os.path.join(output_dir_hydra, f"Time_Lum_C_{mc.T_start}.svg"))
    plt.close(fig)

def TL_donors_iso(cfg: DictConfig, x_ax, e_ratio) -> None:
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    #output dirs
    output_dir_local = os.path.join(os.getcwd(), f"results/lifetime/isoTL/")
    output_dir_hydra = os.path.join(PROJECT_ROOT, f"results/lifetime/isoTL/")
    os.makedirs(output_dir_local, exist_ok=True)
    os.makedirs(output_dir_hydra, exist_ok=True)

    colors = plt.cm.rainbow(np.linspace(0, 1, e_ratio.shape[-1]))
    plt.clf()
    # Create two-subplot figure: 1 row, 2 columns
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5), sharey=False)
    for j in range(e_ratio.shape[-2]):
        for i in range(e_ratio.shape[-1]):
            idx = np.argmin(x_ax[:,j,i]!=0)
            ns = sim.analy_TL_iso(cfg,x_ax[:idx,j,i],i)

            # Plot unnormalized on the first subplot
            ax1.plot(
                x_ax[:idx,j,i], e_ratio[:idx,j,i],
                color=colors[i], linewidth=0.5,
                label=f"rho'={mc.rho_prime[i]}",
                linestyle="--"
            )
            ax1.plot(x_ax[:idx,j,i],ns, color='black', linewidth=0.5, label=f"Analytical with rho_prime={mc.rho_prime[i]}", linestyle='--')
    # Unnormalized plot
    ax1.set_xscale('log')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal")
    ax1.set_title("Unnormalized")

    ax1.legend()

    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir_local, f"Time_Donor_rho_{mc.rho_prime}.svg"))
    plt.savefig(os.path.join(output_dir_hydra, f"Time_Donor_rho_{mc.rho_prime}.svg"))
    plt.close(fig)

def OSL_A(cfg:DictConfig,Lums:list,counts:list) -> None:
    mc = cfg["mc"]
    phys = cfg["phys"]
    #output dirs
    output_dir_local = os.path.join(os.getcwd(), "results/figs/OSL_A")
    output_dir_hydra = os.path.join(PROJECT_ROOT, "results/figs/OSL_A")
    
    os.makedirs(output_dir_local, exist_ok=True)
    os.makedirs(output_dir_hydra, exist_ok=True)

    runs = len(Lums)
    colors = plt.cm.rainbow(np.linspace(0, 1, runs))
    plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharey=False)
    axs = axs.flatten()
    for i in range(len(Lums)):
        lum_across_sims = np.mean(Lums[i], axis=1)
        n_steps = Lums[i].shape[0]  
        x_ax = np.linspace(0, mc.duration, n_steps)
        
        #` Generate an array of colors using the "rainbow" colormap
        axs[i].plot(x_ax, lum_across_sims,
                 color=colors[i], linewidth=0.5, 
                 label=f"A: {phys.A[i]}", 
                 linestyle='--')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Intensity (au)")
        axs[i].legend(loc="best")
    fig.tight_layout()
        
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir_local, f"A{phys.A}.svg"))
    plt.savefig(os.path.join(output_dir_hydra, f"A{phys.A}.svg"))
    plt.show()

def CW_IRSL(cfg:DictConfig,Lums:list,counts:list) -> None:
    mc = cfg["mc"]
    phys = cfg["phys"]
    #output dirs
    output_dir_local = os.path.join(os.getcwd(), "results/figs/CW_IRSL")
    output_dir_hydra = os.path.join(PROJECT_ROOT, "results/figs/CW_IRSL")
    
    os.makedirs(output_dir_local, exist_ok=True)
    os.makedirs(output_dir_hydra, exist_ok=True)


    runs = len(Lums)
    colors = plt.cm.rainbow(np.linspace(0, 1, runs))
    plt.clf()
    for i in range(len(Lums)):
        lum_across_sims = np.mean(Lums[i], axis=1)
        normalized_lum = lum_across_sims/np.max(lum_across_sims)
        n_steps = Lums[i].shape[0]  
        x_ax = np.linspace(0, mc.duration, n_steps)

        #` Generate an array of colors using the "rainbow" colormap
        plt.plot(x_ax, normalized_lum,
                 color=colors[i], linewidth=0.5, 
                 label=f"A={phys.A[i]}", 
                 linestyle='--')
    plt.xlabel("IR-stimulation time, s")
    plt.ylabel("Normalized CW-IRSL")
    plt.legend(bbox_to_anchor=(0.5, 0.9), loc='upper right')
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir_local, f"CW-IRSL_A_{phys.A}.svg"))
    plt.savefig(os.path.join(output_dir_hydra, f"CW-IRSL_A_{phys.A}.svg"))
    plt.show()

def TL_iso_bg(cfg:DictConfig,x_ax,lum,e_ratio) -> None:
    mc = cfg.exp_type_fp
    phys = cfg.physics_fp
    #output dirs
    output_dir_hydra = os.path.join(os.getcwd(), "results/figs/BG")
    output_dir_local = os.path.join(PROJECT_ROOT, "results/figs/BG")
    os.makedirs(output_dir_local, exist_ok=True)
    os.makedirs(output_dir_hydra, exist_ok=True)

    runs = e_ratio.shape[-1]
    colors = plt.cm.rainbow(np.linspace(0, 1, runs))
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    window_size = 50
    for j in range(e_ratio.shape[-1]):
        for i in range(e_ratio.shape[-2]):
            #e_ratio_sims = np.mean(e_ratio[:,:,i], axis=1) #mean across nsims for each temperature
            #e_ratio_sims_normalized = e_ratio_sims / np.max(e_ratio_sims)
            # Suppose lum_across_sims is your raw signal array
            
            idx = np.argmin(x_ax[:,i,j]!=0)
            smoothed_e = running_mean(e_ratio[:idx,i,j], window_size=window_size)
            x_smoothed = x_ax[:idx,i,j][(window_size - 1) // 2 : -(window_size // 2)]
            plt.plot(x_smoothed[:idx], smoothed_e[:idx],color=colors[j], linewidth=0.5, label=f'Total number of electron traps{mc.N_e[j]}', linestyle='--')
    ax.set_xscale('log')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1e13)
    #plt.text(0.6,0.9,"Heating Rate ($^{\circ}C.s^{-1}$)",
    #transform=plt.gca().transAxes,   # interpret x,y as axes coords
    #fontsize=12,)
    
    plt.xlabel("Log time ")
    plt.ylabel("Percent of filled traps")
    plt.legend(bbox_to_anchor=(0.7, 0.9), loc='upper left')
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir_local, f"D{phys.D}N{mc.N_e}.svg"))
    plt.savefig(os.path.join(output_dir_hydra, f"D{phys.D}N{mc.N_e}.svg"))
    
def main(cfg: DictConfig,x_ax,lum_sec) -> None:
    mc = cfg["exp_type_fp"]
    if mc.exp_type == "TL":
        print("Plotting for temperature loop")
        TL_temp(cfg,x_ax,lum_sec)
        #TL_lum_rhoEs(configs,lums,counts)
        #TL_lum_iso(configs,lums,counts)
        #OSL_A(configs,lums,counts)
    elif mc.exp_type == "iso":
        TL_lum_iso(cfg,x_ax,lum_sec)
    elif mc.exp_type == "optic":
        OSL_A(cfg,lums,counts)
        CW_IRSL(cfg,lums,counts)
    print("Done!")

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main_solo(cfg: DictConfig) -> None:
    print("Starting simulation")
    x_ax, lum, e_ratio = sim.main(cfg)
    mc = cfg.exp_type_fp
    if mc.exp_type == "TL":
        print("Plotting for temperature loop")
        TL_temp(cfg,x_ax,lum)
        #TL_lum_rhoEs(configs,lums,counts)
        #TL_lum_iso(configs,lums,counts)
        #OSL_A(configs,lums,counts)
    elif mc.exp_type == "iso":
        #TL_lum_iso(cfg,x_ax,lum)
        TL_donors_iso(cfg,x_ax,e_ratio)
    elif mc.exp_type == "optic":
        OSL_A(configs,lums,counts)
        CW_IRSL(configs,lums,counts)
    elif mc.exp_type == "background":
        print("Plotting for background radiation")
        TL_iso_bg(cfg,x_ax, lum,e_ratio)
    print("Done!")
    
if __name__ == "__main__":
    main_solo()