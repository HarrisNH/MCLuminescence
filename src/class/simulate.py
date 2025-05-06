# simulate.py
from __future__ import annotations
import numpy as np
from omegaconf import DictConfig
from hydra import main
from engine import initialize_runs
from tl_trap_lab import  TLTrapSim
from paths import CONFIG_DIR, PROJECT_ROOT
import pandas as pd

@main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def simulate(cfg: DictConfig):
    """
    Pure Monte-Carlo driver that returns x_ax, Lum, electron_ratio, configs.
    
    Command-line overrides:
      - duration      : maximum simulation time [s]
      - e_ratio_start : initial fill fraction [0–1]
      - dose_rate     : dose rate D [Gy/s] (set to 0 to disable filling)
      - T_rate        : heating rate [°C/s] (set to 0 to disable heating)
    """
    configs = initialize_runs(cfg)
    n_runs = len(configs)
    mc0    = configs[0].exp_type_fp
    phys  = configs[0].physics_fp
    steps  = int(mc0.steps)
    sims   = int(mc0.sims)
    
    # Pre-allocate outputs
    x_ax           = np.zeros((steps, sims, n_runs))
    Lum            = np.zeros((steps, sims, n_runs))
    electron_ratio = np.zeros((steps, sims, n_runs))
    tag = cfg.get("tag", "")        # CLI:  tag=mylabel

    # Loop over sweep points
    for run_idx, run_cfg in configs.items():
        mc    = configs[run_idx].exp_type_fp
        phys  = configs[run_idx].physics_fp
        # Read overrides (or default 0)
        duration      = float(mc.get("duration", 0.0))
        e_ratio_start = float(mc.get("e_ratio_start", 0.0))
        dose_rate     = float(phys.get("dose_rate", 0.0))
        T_rate        = float(mc.get("T_rate", 0.0))
        MAX_dT = float(mc.get("max_dt", 0.5))
        dt_cap = MAX_dT / T_rate if T_rate else 1e20
        for j in range(sims):
            # Re-initialize simulator for each sim to reset RNG, caches, etc.
            sim = TLTrapSim(run_cfg, e_ratio_start=e_ratio_start)
            t_cur = 0.0
            i = 0
            while t_cur <= duration:
                # 1) Update lifetimes at current temperature
                T0 = getattr(run_cfg.exp_type_fp, "T_start", 0)
                T_now = T0 + T_rate * t_cur+273.15 #To Kelvin
                sim._update_lifetimes(T_now)

                # 2) Draw next filling vs recombination time
                dt_fill  = sim._filling_time(dose_rate)
                dt_recomb = np.min(sim._recomb_wait) if sim._recomb_wait.any() else dt_fill
                dt = min(dt_fill, dt_recomb, dt_cap)

                # 3) Step time
                t_cur += dt
                x_ax[i, j, run_idx] = t_cur

                temp_prev = T0 + T_rate * (t_cur - dt)
                temp_now  = T0 + T_rate * t_cur
                dT        = temp_now - temp_prev if T_rate else dt   # use dT when heating, dt otherwise
                # 4) Perform event
                if dt == dt_fill:
                    # filling event
                    sim.box.add_electron()
                    event = 0

                elif dt == dt_recomb:
                    # recombination event
                    e_idx = int(np.argmin(sim._recomb_wait))
                    _, nearest = sim.box.nearest()
                    h_idx = int(nearest[e_idx])
                    sim.box.remove_pair(e_idx, h_idx)
                    event = 1
                else:
                    # No event
                    event = 0
                Lum[i, j, run_idx] = event

                # 5) Record filled‐trap fraction
                electron_ratio[i, j, run_idx] = sim.box.electrons.shape[0] / sim.mc.N_e
                i += 1
                # 6) Stop if we’ve passed the duration
                if duration and t_cur >= duration:
                    break

    # Return to Hydra; downstream plots can unpack these four objects
    steps, sims, runs = Lum.shape          # example array

    # 1) reorder axes  →  (runs, sims, steps)
    lum_rs = Lum.transpose(2, 1, 0)
    e_ratio_rs = electron_ratio.transpose(2, 1, 0)
    df = (
        pd.DataFrame({
            "run":  np.repeat(np.arange(runs), sims * steps),
            "sim":   np.tile(np.repeat(np.arange(sims), steps), runs),
            "step":   np.tile(np.arange(steps), runs * sims),
            "lum":   lum_rs.ravel(),
            "electron_ratio": e_ratio_rs.ravel()
        })
    )

    df.to_csv(f"{PROJECT_ROOT}/results/simulations/exp_{tag}.csv", index=False)

    return x_ax, Lum, electron_ratio, configs

if __name__ == "__main__":
    simulate()