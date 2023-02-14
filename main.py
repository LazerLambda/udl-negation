"""Main Entry Point for Package.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import hydra
from neg_udl.Experiment import Experiment
from neg_udl.NegDataOnlyExperiment import NegDataOnlyExperiment
from omegaconf import DictConfig

from dotenv import load_dotenv
load_dotenv()

@hydra.main(version_base=None, config_path="neg_udl/config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    if cfg.experiment == "NegDataOnlyExperiment":
        experiment: NegDataOnlyExperiment = NegDataOnlyExperiment(
            name=cfg.name,
            model_checkpoint=cfg.model.name,
            dataset_config=dict(cfg.data),
            seed=cfg.seed,
            num_epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            model_target_path=cfg.model.target_path,
            freeze_layers=(cfg.model.freeze_lower, cfg.model.freeze_upper),
            model_tmp_path=cfg.model.tmp_path
        )
        experiment.prepare_dataset()
        experiment.run()

if __name__ == "__main__":
    run_experiment()