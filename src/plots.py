
import os
from paths import PROJECT_ROOT,CONFIG_DIR
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import matplotlib.pyplot as plt
import sim as sim
import pandas as pd


def time_lum_isothermal(cfg: DictConfig) -> None:
    Lum, count = sim.TL_iso(cfg)
    mc = cfg.exp_type
    t = np.arange(0, mc.duration, mc.dt)

    output_dir = os.path.join(os.getcwd(), "results/figs/isoTL")
    os.makedirs(output_dir, exist_ok=True)

    # Plot individual simulations with label only for the first one
    #normalized_lum = Lum/np.max(Lum,axis=0)

    #Calculate mean luminescence rate per second
    # Number of time steps per second
    steps_per_second = int(1 / mc.dt) 
    # Compute mean luminescence in each full second
    mean_lum_per_second = np.array([
        np.mean(Lum[i:i+steps_per_second], axis=0)  
        for i in range(0, len(Lum), steps_per_second)
    ])  
    normalized_mean_lum_per_second = mean_lum_per_second/np.max(mean_lum_per_second,axis=0)
    x_ax = np.arange(0, len(mean_lum_per_second), 1)
    plt.plot(x_ax, normalized_mean_lum_per_second[:, 0], color='gray', alpha=0.3, linestyle = ":", linewidth=0.5, label=f'Mean lum per second({mc.sims} simulations)')

    if mc.sims > 1:
        plt.plot(x_ax, normalized_mean_lum_per_second[:, 1:], color='gray', alpha=0.3, linestyle = ":")
    
    plt.plot(x_ax, np.mean(normalized_mean_lum_per_second, axis=1), color='blue', linewidth=0.5, label='Mean', linestyle='--')
    
    #plt.plot(np.arange(0, len(mean_lum_per_second), 1), mean_lum_per_second, color='green', linewidth=0.5, label='Mean lum per second', linestyle='--')
    plt.text(0.7,0.7,f"Configs: dt: {mc.dt}\n d_dist: {mc.dr_prime}",transform=plt.gca().transAxes,   # interpret x,y as axes coords
    fontsize=12,
    verticalalignment='top')
    plt.xlabel("Time (s)")
    plt.ylabel(f"Normalized isothermal signal at {mc.T_start} degrees")
    plt.title("Lum(time)")
    plt.legend()
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir, f"Time_Lum_d_dist{mc.dr_prime}dt{mc.dt}.svg"))
    plt.savefig(os.path.join(PROJECT_ROOT, f"results/figs/isoTL/Time_Lum_dt{mc.dt}d_dist{mc.dr_prime}.svg"))

def time_lum_non_isothermal(cfg: DictConfig) -> None:
    Lum, count,steps = sim.TL_non_iso(cfg)
    mc = cfg.exp_type

    output_dir = os.path.join(os.getcwd(), "results/figs/TL")
    os.makedirs(output_dir, exist_ok=True)

    #Normalize and mean across simulations
    #Lum = Lum/mc.N0 #np.max(mean_lum_per_second,axis=0)
    lum_across_sims = np.mean(Lum, axis=1)
    
    duration = (mc.T_end-mc.T_start)/np.array(mc.T_rate)
    n_steps = (duration/mc.dt).astype(int)
    
    x_ax = [np.linspace(mc.T_start, mc.T_end, int(n)) for n in n_steps]

    n_colors = len(n_steps)
    # Generate an array of colors using the "rainbow" colormap
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    for i,n in enumerate(n_steps):
        plt.plot(x_ax[i], lum_across_sims[:n_steps[i],i],color=colors[i], linewidth=0.5, label=f'{mc.T_rate[i]}', linestyle='--')
    plt.text(0.6,0.9,"Heating Rate ($^{\circ}C.s^{-1}$)",
    transform=plt.gca().transAxes,   # interpret x,y as axes coords
    fontsize=12,)
    
    plt.xlabel("Temperature ($^{\circ}C$)")
    plt.ylabel("Intensity ($^{\circ}C^{-1}$)")
    plt.legend(bbox_to_anchor=(0.7, 0.9), loc='upper left')
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir, f"Temp_LumdT{mc.T_rate}.svg"))
    plt.savefig(os.path.join(PROJECT_ROOT, f"results/figs/TL/Temp_LumdTemp{mc.T_rate}.svg"))

def lum_TL_rhoEs(cfg:DictConfig):
    output_dir = os.path.join(os.getcwd(), "results/figs/TL_rhoEs")
    os.makedirs(output_dir, exist_ok=True)

    mc = cfg.exp_type
    phys = cfg.physics

    steps = (mc.T_end-mc.T_start)/(np.array(mc.T_rate)*mc.dt)
    x_ax = np.linspace(mc.T_start, mc.T_end, int(steps))
    runs = len(phys.rho_prime)
    colors = plt.cm.rainbow(np.linspace(0, 1, runs))
    for i in range(runs):
        rho_prime = phys.rho_prime[i]
        E = phys.E[i]
        s = phys.s_tun[i]
        Lum,count,rs = sim.TL_rhoEs(cfg,rho_prime,E,s)
        plt.plot(
        x_ax,
        np.mean(Lum, axis=1),
        label=f"$\\rho'$: {rho_prime:.0e}, E={E:.0e}eV, s_tun={s:.0e}$s^-1$",
        color=colors[i]
        )
    plt.xlabel("Temperature ($^{\circ}C$)")
    plt.ylabel("Intensity ($^{\circ}C^{-1}$)")
    plt.legend(bbox_to_anchor=(0.5, 0.9), loc='upper left')
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir, f"rhoEs{runs}.svg"))
    plt.savefig(os.path.join(PROJECT_ROOT, f"results/figs/TL_rhoEs/rhoEs{runs}.svg"))

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.exp_type.name == "isothermal":
        print("Running isothermal simulation")
        time_lum_isothermal(cfg)
    elif cfg.exp_type.name == "non_isothermal":
        print("Running non-isothermal simulation")
        time_lum_non_isothermal(cfg)
    elif cfg.exp_type.name == "TL_rhoEs":
        print("Running TL_rhoEs simulation")
        lum_TL_rhoEs(cfg)
    print("Done!")

if __name__ == "__main__":
    main()