import hydra
from omegaconf import DictConfig

from density_annotator import DensityAnnotator


@hydra.main(config_path="./conf", config_name="annotator")
def run_annotator(cfg: DictConfig):
    annotator = DensityAnnotator(cfg)
    annotator.run()


if __name__ == "__main__":
    run_annotator()
