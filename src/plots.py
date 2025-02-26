
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

    output_dir = os.path.join(os.getcwd(), "results/figs")
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
    plt.savefig(os.path.join(PROJECT_ROOT, f"results/figs/Time_Lum_dt{mc.dt}d_dist{mc.dr_prime}.svg"))

def time_lum_non_isothermal(cfg: DictConfig) -> None:
    Lum, count,steps = sim.TL_non_iso(cfg)
    mc = cfg.exp_type

    output_dir = os.path.join(os.getcwd(), "results/figs")
    os.makedirs(output_dir, exist_ok=True)

    # Plot individual simulations with label only for the first one
    #normalized_lum = Lum/np.max(Lum,axis=0)



    #Calculate mean luminescence rate per second
    # Number of time steps per second
    steps_per_kelvin = int(1 / mc.dt) * mc.T_rate # Steps/sec*seconds/kelvin= steps/kelvin 
    # Compute mean luminescence in each full second
    mean_lum_per_second = np.array([
        np.mean(Lum[i:i+steps_per_kelvin], axis=0)  
        for i in range(0, len(Lum), steps_per_kelvin)
    ])  
    normalized_mean_lum_per_second = mean_lum_per_second/1#np.max(mean_lum_per_second,axis=0)
    x_ax = np.linspace(mc.T_start, mc.T_end, len(mean_lum_per_second))


    plt.plot(x_ax, normalized_mean_lum_per_second[:, 0], color='gray', alpha=0.3, linestyle = ":", linewidth=0.5, label=f'Mean lum per second({mc.sims} simulations) (Heat rate: {mc.T_rate})')

    if mc.sims > 1:
        plt.plot(x_ax, normalized_mean_lum_per_second[:, 1:], color='gray', alpha=0.3, linestyle = ":")
    
    plt.plot(x_ax, np.mean(normalized_mean_lum_per_second, axis=1), color='blue', linewidth=0.5, label='Mean', linestyle='--')
    
    #plt.plot(t[0:-(window_size-1)], np.mean(rolling_mean, axis=1), color='red', linewidth=0.5, label=f'Rolling Mean size {window_size}', linestyle='--')
    #plt.plot(np.arange(0, len(mean_lum_per_second), 1), mean_lum_per_second, color='green', linewidth=0.5, label='Mean lum per second', linestyle='--')
    plt.text(0.7,0.7,f"Configs: dt: {mc.dt}\n d_dist: {mc.dr_prime}",transform=plt.gca().transAxes,   # interpret x,y as axes coords
    fontsize=12,
    verticalalignment='top')
    plt.xlabel("Temperature (C)")
    plt.ylabel(f"Normalized nonisothermal signal")
    plt.title("Lum(time)")
    plt.legend()
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir, f"Time_Non_Lum_d_dist{mc.dr_prime}dt{mc.dt}dTemp{mc.T_rate}.svg"))
    plt.savefig(os.path.join(PROJECT_ROOT, f"results/figs/Time_Non_Lum_dt{mc.dt}d_dist{mc.dr_prime}dTemp{mc.T_rate}.svg"))

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.exp_type.name == "isothermal":
        print("Running isothermal simulation")
        time_lum_isothermal(cfg)
    elif cfg.exp_type.name == "non_isothermal":
        print("Running non-isothermal simulation")
        time_lum_non_isothermal(cfg)
    print("Done!")