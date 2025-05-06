from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

# -----------------------------------------------------------------------------
# 1.  Configuration sweep helper (unchanged apart from type hints)
# -----------------------------------------------------------------------------

def initialize_runs(cfg: DictConfig) -> Dict[int, DictConfig]:
    """Expand list‑valued fields in *cfg* into one run‑config per cartesian point."""
    configs = {
        "exp_type_fp": cfg.exp_type_fp,
        "physics_fp" : cfg.physics_fp,
    }

    lengths = [len(v) for sub in configs.values() for v in sub.values() if isinstance(v, ListConfig)]
    max_length = max(lengths) if lengths else 1

    runs: Dict[int, DictConfig] = {}
    for i in range(max_length):
        run_dict: Dict[str, Dict] = {}
        for sub_name, sub in configs.items():
            sub_run: Dict[str, object] = {}
            for k, v in sub.items():
                if isinstance(v, ListConfig):
                    lst = list(v)
                    if max_length % len(lst) != 0:
                        raise ValueError(f"max_length {max_length} is not divisible by len({sub_name}.{k})={len(lst)}")
                    sub_run[k] = (lst * (max_length // len(lst)))[i]
                else:
                    sub_run[k] = v
            run_dict[sub_name] = OmegaConf.create(sub_run)
        runs[i] = OmegaConf.create(run_dict)
    return runs

# -----------------------------------------------------------------------------
# 2.  Immutable physics parameters + rate formulas
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Physics:
    """A thin value‑object around your *physics_fp* namespace."""

    alpha  : float  # decay length for tunnelling (1/Å)
    b      : float  # localisation prefactor (s⁻¹)
    s      : float  # conduction‑band prefactor (s⁻¹)
    E_cb   : float  # eV
    E_loc_1: float  # eV
    E_loc_2: float  # eV
    D0     : float  # Gy⁻¹ s⁻¹ – dose‑rate scale (used in filling)
    Retrap : float  # probability of getting the *shallow* localisation channel
    name: str | None = None
    D: float | None = None
    rho_trap: float | None = None
    # fundamental constants
    k_b: float = 8.617333262145e-5  # eV K⁻¹ (Boltzmann)

    # ---------------------------------------------------------------------
    #  Rate helpers
    # ---------------------------------------------------------------------
    def rate_cb(self, T: float) -> float:
        """Conduction‑band emission rate (scalar)."""
        return self.s * np.exp(-self.E_cb / (self.k_b * T))

    def rate_tunnel(self, d: np.ndarray, T: float) -> np.ndarray:
        """Element‑wise tunnelling rate for an array of distances *d* (Å)."""
        # choose localisation depth per *distance* element once for repeatability
        E_loc = np.where(np.random.rand(*d.shape) < self.Retrap, self.E_loc_2, self.E_loc_1)
        return self.b * np.exp(-E_loc / (self.k_b * T) - self.alpha * d)

    def lifetime(self, d: np.ndarray, T: float) -> np.ndarray:
        """Combined lifetime for nearest electron–hole distances *d*."""
        return 1.0 / (self.rate_cb(T) + self.rate_tunnel(d, T))

# -----------------------------------------------------------------------------
# 3.  The simulation box (electrons, holes, distance bookkeeping)
# -----------------------------------------------------------------------------

@dataclass
class Box:
    """Tracks electron / hole coordinates and keeps a lazy distance matrix."""

    L: float  # nm (core cube side)
    W: float
    H: float
    boundary_factor: float

    electrons: np.ndarray = field(init=False, repr=False)
    holes    : np.ndarray = field(init=False, repr=False)
    _d       : np.ndarray | None = field(default=None, init=False, repr=False)
    _min_d   : np.ndarray | None = field(default=None, init=False, repr=False)
    _nearest : np.ndarray | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    @property
    def core_shape(self) -> Tuple[float, float, float]:
        return self.L, self.W, self.H

    @property
    def boundary_shape(self) -> Tuple[float, float, float]:
        bf = self.boundary_factor
        return self.L * bf, self.W * bf, self.H * bf
    
    # ------------------------------------------------------------------
    # distance helpers
    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Full recomputation of the distance matrix and nearest arrays."""
        e = self.electrons[:, None, :]
        h = self.holes[None, :, :]
        self._d       = np.linalg.norm(e - h, axis=2)
        self._min_d   = np.min(self._d, axis=1)
        self._nearest = np.argmin(self._d, axis=1)

    # ------------------------------------------------------------------
    # initial seeding
    # ------------------------------------------------------------------
    def seed(self, n_e: int, n_h: int) -> None:
        self.electrons = np.random.rand(n_e, 3) * self.core_shape
        # extra boundary holes to keep density constant in enlargened volume
        total_h = int(n_h * self.boundary_factor**3)
        self.holes     = np.random.rand(total_h, 3) * self.boundary_shape
        self._d = None  # invalidate cache
    # ------------------------------------------------------------------
    # coordinate mutations
    # ------------------------------------------------------------------
    def add_electron(self) -> None:
        """Append one (electron, hole) pair and update caches incrementally."""
        new_e = np.random.rand(1, 3) * self.core_shape
        new_h = np.random.rand(1, 3) * self.boundary_shape

        old_holes = self.holes.copy()
        self.electrons = np.vstack([self.electrons, new_e])
        self.holes     = np.vstack([self.holes,     new_h])

        if self._d is None:
            self._rebuild()
            return

        # --- incremental update ----------------------------------------
        new_row = np.linalg.norm(new_e - old_holes, axis=1)          # e→all h
        new_col = np.linalg.norm(self.electrons - new_h, axis=1)[:, None]  # h←all e
        self._d = np.vstack([self._d, new_row])
        self._d = np.hstack([self._d, new_col])
        self._min_d   = np.append(self._min_d, np.min(new_row))
        self._nearest = np.append(self._nearest, np.argmin(new_row))

    def remove_pair(self, e_idx: int, h_idx: int) -> None:
        self.electrons = np.delete(self.electrons, e_idx, axis=0)
        self.holes     = np.delete(self.holes,     h_idx, axis=0)

        if self._d is None:
            return  # nothing cached

        # delete corresponding row & column
        self._d      = np.delete(self._d, e_idx, axis=0)
        self._d      = np.delete(self._d, h_idx, axis=1)
        self._min_d  = np.delete(self._min_d, e_idx)
        self._nearest= np.delete(self._nearest, e_idx)

        # shift hole indices that were after the removed hole
        self._nearest[self._nearest > h_idx] -= 1

        # electrons that lost their nearest hole need an update
        mask = self._nearest == h_idx
        if mask.any():
            slice_d = self._d[mask]                  # sub‑matrix e×(H−1)
            self._min_d[mask]   = np.min(slice_d, axis=1)
            self._nearest[mask] = np.argmin(slice_d, axis=1)

    # ------------------------------------------------------------------
    # distance matrix & nearest‑hole helper
    # ------------------------------------------------------------------
    def distances(self) -> np.ndarray:
        if self._d is None:
            self._rebuild()
        return self._d

    def nearest(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._min_d is None:
            self._rebuild()
        return self._min_d, self._nearest
