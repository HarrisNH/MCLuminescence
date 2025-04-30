import numpy as np
from  scipy.optimize import curve_fit

# --- model -------------------------------------------------------------
def sat_exp(D, Lsat, D0):
    """Saturating-exponential growth curve."""
    return Lsat*(1 - np.exp(-D/D0))

# --- main fitting helper ----------------------------------------------
def estimate_D0(D, L, sigma=None):
    """
    Fit a saturating exponential and return D0 (+ 1-σ error).

    Parameters
    ----------
    D     : 1-D array-like  – doses (Gy)
    L     : 1-D array-like  – luminescence signal
    sigma : 1-D array-like, optional – 1-σ uncertainties on L

    Returns
    -------
    (Lsat, D0), (σ_Lsat, σ_D0)
    """
    D = np.asarray(D, float)
    L = np.asarray(L, float)

    # crude start guess: max-signal and ~70 % of max-dose
    p0 = (L.max(), 0.7*D.max())

    popt, pcov = curve_fit(
            sat_exp, D, L,
            p0=(1.5, 300),          # << better initial values
            bounds=([0,   0],       #    keep parameters positive
                    [10, 5000]),    #    and within a plausible range
            sigma=sigma,
            absolute_sigma=sigma is not None
    )
    perr = np.sqrt(np.diag(pcov))          # 1-σ errors
    return popt, perr


# ----------------- EXAMPLE:  FSM-13 data ------------------------------
if __name__ == "__main__":
    # laboratory regeneration times (s)  → dose (Gy)
    t   = np.array([1086,2172,3257,4343,5429,6515,7600,8686,9772,10858,11944])
    dR  = 0.092                      # Gy s⁻¹
    D   = t * dR
    L   = np.array([0.03,0.07,0.15,0.28,0.43,0.59,0.74,0.88,1.03,1.16,1.33])

    (Lsat, D0), (sLsat, sD0) = estimate_D0(D, L)
    print(f"Lsat = {Lsat:.3f} ± {sLsat:.3f}")
    print(f"D0   = {D0:.1f} ± {sD0:.1f} Gy")