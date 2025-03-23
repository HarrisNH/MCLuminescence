import os
from paths import PROJECT_ROOT,CONFIG_DIR
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import matplotlib.pyplot as plt
import sim as sim

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    mc = cfg.exp_type
    Lum, count = sim.run_sim(cfg)
    t = np.arange(0, mc.duration, mc.dt)

    output_dir = os.path.join(os.getcwd(), "results/figs")
    os.makedirs(output_dir, exist_ok=True)

    # Plot individual simulations with label only for the first one
    normalized_lum = Lum/np.max(Lum,axis=0)
    plt.plot(t, normalized_lum[:, 0], color='gray', alpha=0.3, label=f"{mc.sims} monte carlo simulations", linestyle = ":")
    if mc.sims > 1:
        plt.plot(t, normalized_lum[:, 1:], color='gray', alpha=0.3, linestyle = ":")
    
    plt.plot(t, np.mean(normalized_lum, axis=1), color='blue', linewidth=0.5, label='Mean', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel(f"Normalized isothermal signal at {mc.T} degrees")
    plt.title("Lum(time)")
    plt.legend()
    # Save your plot in the new directory
    plt.savefig(os.path.join(output_dir, "Time_Lum.svg"))
    plt.savefig(os.path.join(PROJECT_ROOT, "results/figs/Time_Lum.svg"))

if __name__ == "__main__":
    main()


