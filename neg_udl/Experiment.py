"""Main Experiment Class.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import logging
import os
import pathlib
import re
import typing
from typing import Any, Optional, Union

import life_after_bert
import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling as DataCollator4LM
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
            data_collator: str,
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
        :param data_collator: String of data collator that will be used in this experiment.
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
        frozen_layer: Union[list, str] = "No frozen layers"
        if bool(freeze_layers):
            frozen_layer = self._freeze_layers(
                freeze_layers[0],
                freeze_layers[1])

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=True,
            max_len=512)

        self.optimizer: AdamW = AdamW(self.model.parameters(), lr=lr)

        self.dataset_config: dict = dataset_config
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size

        self.path_target: str = os.path.join(
            model_target_path,
            'N-' + self.model.config._name_or_path + '.pt')
        pathlib.Path(model_target_path).mkdir(parents=True, exist_ok=True)

        self.model_tmp_path: str = os.path.join(
            model_tmp_path,
            'TMP-N-' + self.model.config._name_or_path + '.pt')
        pathlib.Path(model_tmp_path).mkdir(parents=True, exist_ok=True)

        # Set Data Collator
        self.data_collator: Any = -1
        if data_collator == 'DataCollatorForTokenClassification':
            self.data_collator = DataCollator4TC(self.tokenizer)
            logging.info(f"Initialized: {data_collator}")
        if data_collator == 'DataCollatorForLanguageModeling':
            self.data_collator = DataCollator4LM(self.tokenizer)
            logging.info(f"Initialized: {data_collator}")
        if self.data_collator == -1:
            raise Exception("Data Collator must be defined correctly!")

        # Set up antonym negation ds
        hub_dataset: typing.Any = load_dataset("KevinZ/oLMpics", "Antonym_Negation")["test"]
        self.oLMpics_dataset: typing.Any = life_after_bert.MCDataset(
            hub_dataset["stem"],
            hub_dataset["choices"],
            hub_dataset["answerKey"],
            num_choices=2,
            tokenizer=self.tokenizer)

        # Set Up Logging
        wandb.init(config={
            'experiment': name,
            'model checkpoint': self.model_checkpoint,
            'model': self.model.config.to_dict(),
            ##
            'model name': self.model.config._name_or_path,
            'model: attention dropout prob': self.model.config.attention_probs_dropout_prob,
            'model: classifier dropout': self.model.config.classifier_dropout,
            'model: hidden act fun': self.model.config.hidden_act,
            ##
            'optimizer': self.optimizer.state_dict(),
            'epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'frozen layers': f"{freeze_layers[0]}-{freeze_layers[1]}",
            'frozen layers list': frozen_layer
        })
        wandb.run.name = name

    def _freeze_layers(self, begin: int, end: int) -> list:
        """Freeze Layers of Model.

        :param begin: Begin of range for layer freezing.
        :param end: End of range for layer freezing.
        :returns: List of frozen layers.
        """
        regex: str = r"layer\.([0-9]+)\."
        rng: list = list(range(begin, end))
        ret_list: list = []
        for name, param in self.model.named_parameters():
            if re.search(regex, name) is not None:
                layer: int = int(re.search(regex, name).group(1))
                if layer in rng:
                    logging.info(f"FREEZE Layer: {name}")
                    param.requires_grad = False
                    ret_list.append(str(name) + ' - YES')
                    continue
                else:
                    ret_list.append(str(name) + ' - NO')
                    logging.info(f"KEEP Layer: {name}")
                    continue
            ret_list.append(str(name) + ' - NO')
            logging.info(f"KEEP Layer: {name}")
        return ret_list

    def prepare_dataset(self) -> None:
        """Prepare Dataset.

        Must assign a value to `self.dataset`. Use `self.dataset_config` to access
        necessary information about dataset.

        Prepare an instance of a `Dataset` class or subclass that has a 'split' and a
        'valid' split to the variable `self.dataset`. These two keys are required for
        later training.

        :raises: NotImplementedError
        """
        raise NotImplementedError("`prepare_dataset` must be implemented for each experiment.")

    def _eval_test(self, eval_dataloader: Any) -> Optional[float]:
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
            total_loss += n_batch * outputs.loss.item()

            self.model.train()
        return ((total_loss / n) if n != 0 else None)

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
        self.model.to(self.device)

        logging.info("Load Train-Dataset.")
        train_dataloader = DataLoader(
            self.dataset["train"],
            shuffle=True, batch_size=self.batch_size,
            collate_fn=self.data_collator
        )
        logging.info("Train-Dataset loaded.")
        logging.info("Load Test-Dataset.")
        eval_dataloader = DataLoader(
            self.dataset["valid"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator
        )
        logging.info("Test-Dataset loaded.")

        # TODO: Rm self?
        num_training_steps = self.num_epochs * len(train_dataloader)
        self.lr_scheduler: typing.Any = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        eval_total_loss: Optional[float] = self._eval_test(eval_dataloader)
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

        torch.save(
            {
                'epoch': epoch,  # TODO: Rm?
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),  # TODO: Rm?
            },
            self.path_target)
        logging.info('Saved trained model at:' + self.path_target)

        os.remove(self.model_tmp_path)
        logging.info('Training checkpoint deleted.')
