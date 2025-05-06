# tl_trap_simulator_refactored.py
"""Monte‑Carlo thermoluminescence simulator.
* ``Physics``   – immutable constants and helper rate formulas
* ``Box``       – owns the mutable electron / hole clouds and all geometry
* ``TLTrapSim`` – orchestrates the Monte‑Carlo time loop and comparison with
                  laboratory data
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

from engine import Box, Physics, initialize_runs
from paths import PROJECT_ROOT 


class TLTrapSim:
    """Monte‑Carlo engine reproducing the old *sim_lab_* functions."""

    def __init__(self, cfg: DictConfig, *, e_ratio_start: float | None = None):
        self.cfg   = cfg
        self.mc    = cfg.exp_type_fp
        self.phys  = Physics(**cfg.physics_fp)

        # derive initial geometry once
        rho = self.mc.rho_prime * (3/(4*np.pi) * self.phys.alpha**3)
        d   = (self.mc.holes / rho) ** (1/3)
        self.box = Box(d, d, d, self.mc.boundary_factor)

        # seed initial carriers
        e0 = int(self.mc.N_e * (e_ratio_start if e_ratio_start is not None else 0))
        self.box.seed(e0, self.mc.holes)

        # internal caches that change during a run
        self._lifetime: np.ndarray | None = None
        self._recomb_wait: np.ndarray | None = None

    # ------------------------------------------------------------------
    #  Private helpers – rates, filling time, recombination handling
    # ------------------------------------------------------------------
    def _update_lifetimes(self, T: float) -> None:
        d_min, _ = self.box.nearest()
        self._lifetime = self.phys.lifetime(d_min, T)
        self._recomb_wait = np.random.exponential(self._lifetime)

    def _filling_time(self, D: float) -> float:
        e = self.box.electrons.shape[0]
        N = int(self.mc.N_e)
        if e == N or D == 0:
            lam = 1e-20  # mean = 1/λ  → effectively infinity
        else:
            lam = (D / self.phys.D0) * (N - e)
        return np.random.exponential(1/lam) if lam > 0 else 1e20

    # ------------------------------------------------------------------
    #  Functions to simulate the lab experiments
    # ------------------------------------------------------------------
    def TL_lab(self, csv_name: str, *, plot: bool = False) -> float:
        """Replicates *sim_lab_TL_residuals* (temperature‑ramp experiments)."""
        path = Path(PROJECT_ROOT) / "data/processed" / f"{csv_name}.csv"
        lab_cfg = pd.read_csv(path)

        SE: List[float] = []
        ER: List[float] = []
        TSS: List[float] = []
        plot_rows: List[Dict[str, float]] = []

        for k, row in lab_cfg.iterrows():
            # reset simulation state for each lab row
            self.__init__(self.cfg, e_ratio_start=row.e_ratio)
            T_start = row.T_start + 273.15
            T_rate  = (row.T_end - row.T_start) / row.Duration

            self._update_lifetimes(T_start)

            D  = self.phys.D  # dose rate irrelevant here – assumed constant
            tf = self._filling_time(D)

            t_cur = 0.0
            sim_times: List[float] = []
            sim_ratio: List[float] = []

            while t_cur < row.Duration:
                dt_recomb = np.min(self._recomb_wait) if self._recomb_wait.size else tf
                dt        = min(dt_recomb, tf)
                T_now     = T_start + T_rate * (t_cur + dt)
                t_cur    += dt

                if dt == tf:  # filling event
                    self.box.add_electron()
                else:         # recombination event of the *first* electron in line
                    e_idx = int(np.argmin(self._recomb_wait))
                    _, h_idx = self.box.nearest()
                    h_idx = h_idx[e_idx]
                    self.box.remove_pair(e_idx, h_idx)
                # refresh timers after any structural change
                tf = self._filling_time(D)
                self._update_lifetimes(T_now)

                sim_times.append(t_cur)
                sim_ratio.append(self.box.electrons.shape[0] / self.mc.N_e)

            # ------- error metrics vs. lab fill ratio at end of ramp ---------
            err = sim_ratio[-1] - row.Fill
            SE.append(err**2)
            ER.append(abs(err))
            TSS.append((row.Fill - lab_cfg.Fill.mean())**2)

            plot_rows.append(dict(sim=k, time=row.Duration, ratio=sim_ratio[-1]))

        mse = float(np.mean(SE))
        if plot:
            df_plot = pd.DataFrame(plot_rows)
            self._plot_batch(lab_cfg, df_plot, csv_name)
        self._printer(np.mean(ER), mse)
        return mse

    def ISO_lab(self, csv_name: str, *, plot: bool = False) -> float:
        """Replicates *sim_lab_TL_residuals_iso* (isothermal experiments)."""
        path = Path(PROJECT_ROOT) / "data/processed" / f"{csv_name}.csv"
        lab = pd.read_csv(path)

        SE: List[float] = []
        ER: List[float] = []
        TSS: List[float] = []
        plot_rows: List[Dict[str, float]] = []

        for exp_no in sorted(lab.exp_no.unique()):
            sub = lab[lab.exp_no == exp_no]
            self.__init__(self.cfg, e_ratio_start=float(sub.e_ratio.iloc[0]))
            T_iso = float(sub.temp.iloc[0]) + 273.15
            D     = float(sub.dose.iloc[0])
            self._update_lifetimes(T_iso)
            tf = self._filling_time(D)

            t_cur = 0.0
            obs_times = sub.time.to_numpy()
            obs_idx   = 0

            while obs_idx < len(obs_times):
                dt_recomb = np.min(self._recomb_wait) if self._recomb_wait.size else tf
                dt        = min(dt_recomb, tf)
                t_cur    += dt

                if dt == tf:
                    self.box.add_electron()
                else:
                    e_idx = int(np.argmin(self._recomb_wait))
                    _, h_idx = self.box.nearest()
                    h_idx = h_idx[e_idx]
                    self.box.remove_pair(e_idx, h_idx)

                tf = self._filling_time(D)
                self._update_lifetimes(T_iso)

                # whenever we pass an observation time, evaluate error
                while obs_idx < len(obs_times) and t_cur >= obs_times[obs_idx]:
                    fill   = self.box.electrons.shape[0] / self.mc.N_e
                    target = sub[sub.time == obs_times[obs_idx]].e_ratio.iloc[0]
                    err    = fill - target
                    SE.append(err**2)
                    ER.append(abs(err))
                    TSS.append((target - sub.e_ratio.mean())**2)
                    plot_rows.append(dict(sim=exp_no, time=t_cur, ratio=fill))
                    obs_idx += 1

        mse = float(np.mean(SE))
        if plot:
            df_plot = pd.DataFrame(plot_rows)
            self._plot_iso(lab, df_plot, csv_name)
        self._printer(np.mean(ER), mse)
        return mse

    # ------------------------------------------------------------------
    #  Pretty‑printers & plot helpers (mostly unchanged)
    # ------------------------------------------------------------------

    def _printer(self, avgER: float, MSE: float) -> None:
        p = self.cfg.physics_fp
        mc = self.mc
        print(f"absError={avgER:.3e}, MSE={MSE:.3e}  [rho'={mc.rho_prime}, E_cb={p.E_cb}, D0={p.D0}, "
              f"E_loc1={p.E_loc_1}, E_loc2={p.E_loc_2}, s={p.s}, b={p.b}, alpha={p.alpha}, "
              f"holes={mc.holes}, P_retrap={p.Retrap}]")

    # ------- static plotting helpers (identical style as before) ----------

    @staticmethod
    def _plot_batch(lab_cfg: pd.DataFrame, df: pd.DataFrame, label: str) -> None:
        L0 = 1.52
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.viridis
        uniques = sorted(lab_cfg.T_end.unique())
        colors  = {v: cmap(i/len(uniques)) for i, v in enumerate(uniques)}

        for k, row in df.iterrows():
            t_end = lab_cfg.T_end.iloc[int(row.sim)]
            ax.plot(row.time, row.ratio, "o", color=colors[t_end])

        # lab endpoints
        ax.scatter(lab_cfg.Duration, lab_cfg.Fill/L0, c="red", marker="x", zorder=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Filled fraction")
        ax.set_title(f"Batch experiment – {label}")
        plt.tight_layout()
        out = Path(PROJECT_ROOT) / "results/plots/lab" / f"{label}_batch.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)

    @staticmethod
    def _plot_iso(lab_cfg: pd.DataFrame, df: pd.DataFrame, label: str) -> None:
        L0 = 1.52
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for exp_no in lab_cfg.exp_no.unique():
            col = cmap(exp_no)
            sim = df[df.sim == exp_no]
            lab = lab_cfg[lab_cfg.exp_no == exp_no]
            ax.plot(sim.time, sim.ratio, "o", color=col)
            ax.scatter(lab.time, lab.L/L0, marker="x", color=col)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Filled fraction")
        ax.set_title(f"Isothermal experiment – {label}")
        plt.tight_layout()
        out = Path(PROJECT_ROOT) / "results/plots/lab" / f"{label}_iso.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)

