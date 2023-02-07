"""Main Experiment Class.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import os
import re
import typing
from typing import Any

import life_after_bert
import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForTokenClassification as DataCollator4TC
from transformers import get_scheduler

import wandb


class Experiment:
    """Base-Experiment Class.

    Implementation of initialization and training and eval
    loop. Dataset must be initialized in child class.
    """

    dataset: Dataset = []

    def __init__(
            self,
            name: str,
            model_checkpoint: str,
            dataset_config: dict,
            seed: int,
            num_epochs: int,
            batch_size: int,
            lr: float,
            model_target_path: str,
            freeze_layers: tuple = (),
            model_tmp_path: str = './TMP_CHECKPOINT.pt',
            device: str = ""):
        """Instantiate Experiment Class.

        :param name: Name of the experiment.
        :param model_checkpoint: Model checkpoint to be loaded (only Encoder models).
        :param dataset_config: Dictionary including all dataset related configs.
        :param seed: Seed for reproducability.
        :param num_epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        :param lr: Learning rate for optimizer.
        :param model_target_path: Path where trained model will be saved.
        :param freeze_layers: Tuple, (begin, end) of range of layers which will be
            frozen. To use the whole model, use `()` (default).
        :param model_tmp_path: Path where model checkpoints will be saved during
            training.
        :param device: Device on which training will be executed.
        """
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        np.random.seed(seed)

        self.model_checkpoint: str = model_checkpoint
        self.model: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(
            model_checkpoint)
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device: str = device
        self.model.to(self.device)
        if not bool(freeze_layers):
            self._freeze_layers(freeze_layers[0], freeze_layers[1])

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=True)

        self.optimizer: AdamW = AdamW(self.model.parameters(), lr=lr)

        self.dataset_config: dict = dataset_config
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size
        self.model_target_path: str = model_target_path
        self.model_tmp_path: str = model_tmp_path

        # Set up antonym negation ds
        hub_dataset: typing.Any = load_dataset("KevinZ/oLMpics", "Antonym_Negation")["test"]
        self.oLMpics_dataset: typing.Any = life_after_bert.MCDataset(
            hub_dataset["stem"],
            hub_dataset["choices"],
            hub_dataset["answerKey"],
            num_choices=2,
            tokenizer=self.tokenizer)

        # log = logging.getLogger(__name__)
        # Set Up Logging
        wandb.init(config={
            'experiment': name,
            'model checkpoint': self.model_checkpoint,
            'model': self.model.config,
            ##
            'model name': self.model.config._name_or_path,
            'model: attention dropout prob': self.model.config.attention_probs_dropout_prob,
            'model: classifier dropout': self.model.config.classifier_dropout,
            'model: hidden act fun': self.model.config.hidden_act,
            ##
            'loss': str(self.optimizer),
            'epochs': self.num_epochs,
            'frozen layers': f"{freeze_layers[0]}-{freeze_layers[1]}"
        })
        wandb.run.name = name

    def _freeze_layers(self, begin: int, end: int) -> None:
        """Freeze Layers of Model.

        :param begin: Begin of range for layer freezing.
        :param end: End of range for layer freezing.
        """
        regex: str = f"\\.[{begin}-{end}]\\."  # TODO:
        for name, param in self.model.named_parameters():
            if bool(re.search(regex, name)):
                print(name)  # TODO: Remove
                param.requires_grad = False

    def prepare_dataset(self) -> None:
        """Prepare Dataset.

        Must assign a value to `self.dataset`. Use `self.dataset_config` to access
        necessary information about dataset.
        :raises: NotImplementedError
        """
        raise NotImplementedError("`prepare_dataset` must be implemented for each experiment.")

    def _eval_test(self, eval_dataloader: Any) -> float:
        """Evaluate Model on Test Dataset.

        Compute cross-entropy on test dataset.

        :param eval_dataloader: Dataloader on which model will
            be evaluated.
        :returns: Total loss.
        """
        n: int = 0
        total_loss: float = 0
        for batch in eval_dataloader:
            self.model.eval()

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            n_batch: int = len(batch['input_ids'])
            n += n_batch
            total_loss += n_batch * outputs.loss.cpu().item()

            self.model.train()
        total_loss /= n
        return total_loss

    def _eval_antonym_negation(self) -> float:
        """Eval on oLMpics' Antonym Nnegation.

        Compute accuracy on antomy negation dataset according to
        Lialin et al. 2022.

        :returns: Accuracy of evaluation.
        """
        accuracy, _ = life_after_bert.evaluate_encoder(
            self.model,
            self.tokenizer,
            self.oLMpics_dataset,
            device=self.device)
        return accuracy

    def run(self) -> None:
        """Run Experiment."""
        if not bool(self.dataset):
            self.prepare_dataset()

        data_collator: DataCollator4TC = DataCollator4TC(self.tokenizer)

        train_dataloader = DataLoader(
            self.dataset["train"],
            shuffle=True, batch_size=self.batch_size,
            collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            self.dataset["valid"],
            batch_size=self.batch_size,
            collate_fn=data_collator
        )

        # TODO: Rm self?
        num_training_steps = self.num_epochs * len(train_dataloader)
        self.lr_scheduler: typing.Any = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))

        eval_total_loss: float = self._eval_test(eval_dataloader)
        eval_antonym_negation: float = self._eval_antonym_negation()
        wandb.log(
            {
                'total-loss': eval_total_loss,
                'antonym-negation': eval_antonym_negation
            })

        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
            # TODO: Eval
            # - GLUE ?
            eval_total_loss = self._eval_test(eval_dataloader)
            eval_antonym_negation = self._eval_antonym_negation()

            wandb.log(
                {
                    'total-loss': eval_total_loss,
                    'antonym-negation': eval_antonym_negation
                })

            # log.info(total_loss)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },
                self.model_tmp_path)

        path_target: str = os.path.join(
            self.model_target_path,
            'N-' + self.model.config._name_or_path + '.pt')
        torch.save(
            {
                'epoch': epoch,  # TODO: Rm?
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),  # TODO: Rm?
            },
            path_target)
        # log.info('Saved trained model at:' + path_target)

        os.remove(self.model_tmp_path)
        # log.info('Training checkpoint deleted.')
