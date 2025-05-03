import hydra
from omegaconf import DictConfig
from paths import CONFIG_DIR, PROJECT_ROOT
import plots1p as plts
import sim1p as sim
import results as res
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config_fp")
def main(cfg: DictConfig) -> None:

    print("Starting simulation")
    x_ax, lum, e_ratio,configs = sim.main(cfg)
    res.main(cfg,x_ax, lum, e_ratio,configs)
    plts.main(cfg,x_ax, lum, e_ratio,configs)

    print("Done!")
    
if __name__ == "__main__":
    main()
