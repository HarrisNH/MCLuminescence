import hydra
import plots as plts
from omegaconf import DictConfig
from paths import CONFIG_DIR
import sim as sim


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print("Starting simulation")
    plts.main(cfg)
    print("Done!")
    
if __name__ == "__main__":
    main()
