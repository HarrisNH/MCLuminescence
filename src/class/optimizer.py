# optimizer.py
"""
Global optimiser for trap‑model parameters.

Run examples
------------
# train  – run DE for 10 generations and append results to CSV
$ python optimizer.py task=train exp=tl_clbr gens=10

# replay – take the best line from a previous CSV and re‑plot the simulation
$ python optimizer.py task=replay exp=tl_clbr
"""
from __future__ import annotations
import os
from pathlib import Path
from typing  import Tuple, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from hydra import compose, initialize, initialize_config_dir
import hydra
from scipy.optimize import differential_evolution
from engine import initialize_runs, Physics, Box
from tl_trap_lab import initialize_runs, TLTrapSim
from paths import CONFIG_DIR, PROJECT_ROOT   # unchanged helpers

# --------------------------------------------------------------------------
# Hyper‑parameters & command‑line flags (handled by Hydra)
# --------------------------------------------------------------------------

DEFAULT_BOUNDS: List[Tuple[float, float]] = [
    (1e-8, 1e-3),     # rho_prime
    (1.9,  2.4),      # E_cb
    (1.2,  1.7),      # E_loc_1
    (1.0,  1.5),      # E_loc_2
    (1e2,  4e2),      # D0
    (1e12, 1e14),     # s
    (1e10, 1e13),     # b
    (1e9,  5e10),     # alpha
    (1e2,  7.5e2),    # holes
    (0.0,  1.0)       # retrap
]

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def cfg_with_params(base: DictConfig, params: np.ndarray) -> DictConfig:
    """Return a *deep copy* of *base* whose physics/exp fields are replaced."""
    cfg = base.copy()
    (rho_prime, E_cb, E_loc_1, E_loc_2,
     D0, s, b, alpha, holes, retrap) = params

    cfg.exp_type_fp.rho_prime = float(rho_prime)
    cfg.exp_type_fp.holes     = float(holes)
    cfg.physics_fp.E_cb       = float(E_cb)
    cfg.physics_fp.E_loc_1    = float(E_loc_1)
    cfg.physics_fp.E_loc_2    = float(E_loc_2)
    cfg.physics_fp.D0         = float(D0)
    cfg.physics_fp.s          = float(s)
    cfg.physics_fp.b          = float(b)
    cfg.physics_fp.alpha      = float(alpha)
    cfg.physics_fp.Retrap     = float(retrap)
    return cfg


def run_one_sim(cfg: DictConfig, exp: str, PLOT: bool = False) -> float:
    """Return MSE for one parameter set and one experiment label."""
    runs = initialize_runs(cfg)
    sim  = TLTrapSim(runs[0])

    if exp == "iso":
        return sim.ISO_lab("CLBR_IR50_ISO", plot=PLOT)
    elif exp == "tl_clbr":
        return sim.TL_lab("CLBR_IRSL50_0.25KperGy", plot=PLOT)
    elif exp == "tl_fsm-13":
        return sim.TL_lab("FSM-13_IRSL50_0.25KperGy", plot=PLOT)
    else:
        raise ValueError(f"Unknown exp '{exp}'")

def objective(p: np.ndarray,cfg: DictConfig, exp: str) -> float:
    test_cfg = cfg_with_params(cfg, p)
    return run_one_sim(test_cfg, exp)
# --------------------------------------------------------------------------
# Hydra entry‑point
# --------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    """Dispatcher for training or replay."""
    # CLI overrides: task=train|replay, exp=..., gens=...
    task = cfg.get("task", "train")
    exp  = cfg.get("exp",  "tl_clbr")
    gens = int(cfg.get("gens", 7))          # DE maxiter
    pop  = int(cfg.get("pop",  15))         # DE popsize
    works = int(cfg.get("works", 4))       # DE workers

    csv_path = Path(PROJECT_ROOT, f"results/lab_sims/result_{exp}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------- TRAIN mode -------------------------------------------
    if task == "train":
        result = differential_evolution(
            objective, DEFAULT_BOUNDS,
            strategy="randtobest1bin",args=(cfg, exp),
            init="sobol",
            mutation=0.5, recombination=0.3,
            maxiter=gens, popsize=pop,
            tol=1e-7, workers = works, disp=True, polish=True
        )

        # save row to CSV
        cols = [f"param_{i}" for i in range(len(result.x))] + ["mse"]
        df   = pd.DataFrame([list(result.x) + [float(result.fun)]], columns=cols)
        df.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())
        print(f"Saved run to {csv_path}")

        # quick sanity‑plot of best
        best_cfg = cfg_with_params(cfg, result.x)
        run_one_sim(best_cfg, exp, PLOT=True)   # with internal plotting

    # -------------- REPLAY mode ------------------------------------------
    elif task == "replay":
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df        = pd.read_csv(csv_path)
        best_row  = df.loc[df.mse.idxmin()]
        best_p    = best_row.filter(like="param_").values.astype(float)
        best_cfg  = cfg_with_params(cfg, best_p)
        run_one_sim(best_cfg, exp, PLOT=True)   # will plot
        print("Re‑plotted best parameters for exp:", exp)

    else:
        raise ValueError("task must be 'train' or 'replay'")


if __name__ == "__main__":
    main()