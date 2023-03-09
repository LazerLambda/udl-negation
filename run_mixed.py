"""Main Entry Point for Package.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import hydra
from neg_udl.Experiment import Experiment
from neg_udl.MixedExperiment import MixedExperiment
from omegaconf import DictConfig
import os
import torch

from dotenv import load_dotenv
load_dotenv()

@hydra.main(version_base=None, config_path="neg_udl/config/MixedExperiment", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    if cfg.experiment == "MixedExperiment":
        experiment: MixedExperiment = MixedExperiment(
            name=cfg.name,
            model_checkpoint=cfg.model.name,
            dataset_config=dict(cfg.data),
            data_collator=cfg.data_collator,
            seed=cfg.seed,
            num_epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            steps=cfg.training.eval_steps_n,
            eval_steps=cfg.training.eval_steps,
            model_target_path=cfg.model.target_path,
            freeze_layers=(cfg.model.freeze_lower, cfg.model.freeze_upper),
            model_tmp_path=cfg.model.tmp_path
        )
        experiment.prepare_dataset()
        experiment.run()

if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.multiprocessing.set_start_method('spawn')
    run_experiment()
