# plot_runs.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

from paths import PROJECT_ROOT, CONFIG_DIR
from simulate import simulate  # <- the new driver you created


# -------------------------------------------------------------------------
# 0. helpers
# -------------------------------------------------------------------------
def running_mean(a: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = np.ones(k) / k
    return np.convolve(a, kernel, "valid")


def colour_cycle(n: int):
    return iter(plt.cm.rainbow(np.linspace(0, 1, n)))


def out(name: str, subfolder: str, tag: str | None = None) -> Path:
    """
    Build an output path in  results/<subfolder>/.
    If *tag* is supplied it is appended to the filename:
        out("TL_run", "plots", tag="fast") -> TL_run_fast.png
    """
    base = Path(PROJECT_ROOT) / "results" / subfolder
    base.mkdir(parents=True, exist_ok=True)
    fname = f"{name}_{tag}.png" if tag else f"{name}.png"
    return base / fname

def hist_and_smooth(t_axis, events, bin_width=1.0, win_deg=50.0):
    # 1) histogram into integer‑°C bins
    bins  = np.arange(0, t_axis.max() + bin_width, bin_width)
    hist, _ = np.histogram(t_axis, bins=bins, weights=events)
    # 2) convert to intensity per °C
    hist = hist / bin_width
    # 3) boxcar smooth over *win_deg* °C
    k = max(1, int(win_deg / bin_width))
    return running_mean(hist, k=k)
# -------------------------------------------------------------------------

def plot_TL(cfg: DictConfig, ax, x_ax, lum, e_ratio, configs):
    mc      = cfg.exp_type_fp
    runs    = lum.shape[-1]
    colours = colour_cycle(runs)

    max_T = mc.T_end[0]              # 800 °C from YAML
    for r in range(runs):
        run_cfg  = configs[r]
        T_rate   = run_cfg.exp_type_fp.T_rate
        T_start  = run_cfg.exp_type_fp.T_start

        t   = x_ax[:, 0, r]                     # first replica’s time axis
        L   = lum[:, :, r].mean(axis=1)         # mean over replicas
        m   = t > 0                             # strip zero‑padding
        T_axis  = T_start + T_rate * t[m]       # °C axis
        events  = L[m]                          # *same* mask here  ←■■ FIX

        smooth  = hist_and_smooth(T_axis, events)
        ax.plot(np.arange(len(smooth)), smooth,
                color=next(colours), label=f"{T_rate} °C/s")

    ax.set_xlim(0, max_T)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("TL intensity (per °C)")
    ax.legend()


def plot_TL_normalized(cfg: DictConfig, ax, x_ax, lum, e_ratio, configs):
    mc      = cfg.exp_type_fp
    runs    = lum.shape[-1]
    colours = colour_cycle(runs)

    max_T = mc.T_end[0]              # 800 °C from YAML
    for r in range(runs):
        run_cfg  = configs[r]
        T_rate   = run_cfg.exp_type_fp.T_rate
        T_start  = run_cfg.exp_type_fp.T_start

        t   = x_ax[:, 0, r]                     # first replica’s time axis
        L   = lum[:, :, r].mean(axis=1)         # mean over replicas
        m   = t > 0                             # strip zero‑padding
        T_axis  = T_start + T_rate * t[m]       # °C axis
        events  = L[m]                          # *same* mask here  ←■■ FIX

        smooth  = hist_and_smooth(T_axis, events)
        ax.plot(np.arange(len(smooth)), smooth / smooth.sum(),
                color=next(colours), label=f"{T_rate} °C/s")

    ax.set_xlim(0, max_T)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("TL intensity (per °C)")
    ax.legend()

def plot_iso(cfg: DictConfig, ax, x_ax, lum, e_ratio):
    colors = colour_cycle(e_ratio.shape[-1])
    for r in range(e_ratio.shape[-1]):
        run_cfg = cfg[r]
        t_start = run_cfg.exp_type_fp.T_start
        f = running_mean(e_ratio[:, :, r].mean(axis=1), k=10)
        t = running_mean(x_ax[:, 0, r],                 k=10)
        ax.plot(t, f, color=next(colors), lw=0.8,
                label=f"Temp = {t_start} °C")
    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filled fraction")
    ax.legend()


def plot_bg(cfg: DictConfig, ax, x_ax, lum, e_ratio, configs):
    mc   = cfg.exp_type_fp
    phys = cfg.physics_fp
    colors = colour_cycle(e_ratio.shape[-1])
    for r in range(e_ratio.shape[-1]):
        run_cfg = configs[r]
        D_rate  = run_cfg.physics_fp.D
        frac = e_ratio[:, :, r].mean(axis=1)
        t    = x_ax[:, 0, r]
        ax.plot(t, frac, color=next(colors), lw=0.8,
                label=f"D={D_rate} Gy/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filled fraction")
    ax.set_ylim(0, 1)
    ax.legend()


# -------------------------------------------------------------------------
# 2. registry
# -------------------------------------------------------------------------
PLOTTERS: Dict[str, Callable] = {
    "TL":  plot_TL,
    "TL_norm": plot_TL_normalized,
    "iso": plot_iso,
    "bg":  plot_bg,
}


# -------------------------------------------------------------------------
# 3. Hydra entry‑point
# -------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def plot_all(cfg: DictConfig) -> None:
    """
    High‑level entry:

    $ python plot_runs.py plots="TL,iso" duration=1e4
    """
    # run the simulator first (or load from disk if you cached it)
    x_ax, lum, e_ratio, configs = simulate(cfg)  # returns arrays & runs dict
    mc = cfg.exp_type_fp
    
    # which plotters?
    tag = cfg.get("tag", "")        # CLI:  tag=mylabel
    wanted = cfg.get("plots", "TL_norm").split(",")   # comma‑separated list
    for key in wanted:
        if key not in PLOTTERS:
            print(f"[plot_runs] unknown key '{key}', skipping"); continue
        fig, ax = plt.subplots(figsize=(6, 4))
        PLOTTERS[key](cfg, ax, x_ax, lum, e_ratio,configs)
        fig.tight_layout()
        fig.savefig(out(f"{key}_run", "plots", tag))
        plt.close(fig)
        print(f"[plot_runs] wrote {key}_run_{tag}.png")


if __name__ == "__main__":
    plot_all()