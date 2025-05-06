from omegaconf import DictConfig
from engine import initialize_runs
from tl_trap_lab import TLTrapSim
import hydra
from paths import CONFIG_DIR, PROJECT_ROOT

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:
    runs = initialize_runs(cfg)
    sim  = TLTrapSim(runs[0])
    mse  = sim.run_batch("CLBR_IRSL50_0.25KperGy", plot=True)
    print("done")

if __name__ == "__main__":
    main()