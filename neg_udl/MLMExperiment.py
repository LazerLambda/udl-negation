"""MLM-Based Experiment with Overrepresented Negation Data.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import logging
import multiprocessing
from typing import Callable, Dict

import datasets
import torch
from datasets import load_dataset

from .Experiment import Experiment


class MLMExperiment(Experiment):
    """Class to Perform MLM Experiment."""

    blocksize: int = -1

    def _group_texts(self, examples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Concactenate Instances Together.

        Concatenates instances and splits them according to the class
        variable `blocksize`.

        :param examples: Batch of the original dataset.
        :returns: Dictionary
        """
        # Concatenate all texts.
        concatenated_examples: dict = {k: sum(examples[k], []) for k in examples.keys()}
        total_length: int = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // self.blocksize) * self.blocksize
        # Split by chunks of max_len.
        result: dict = {
            k: [t[i:(i + self.blocksize)] for i in range(0, total_length, self.blocksize)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def prepare_dataset(self) -> None:
        """Prepare Dataset for MLM Training.

        Set up dataset, load from .txt file, split, tokenize and
        eventually group data according to `blocksize` parameter.
        Requires 'path', 'test-prop' and 'blocksize' in dataset_config
        dict.
        """
        path: str = self.dataset_config['path']
        test: float = self.dataset_config['test-prop']
        self.blocksize = self.dataset_config['blocksize']
        data = load_dataset(
            "text", data_files=path, sample_by="line")
        train_test = data['train'].train_test_split(test_size=test)
        dataset = datasets.DatasetDict({
            'train': train_test['train'],
            'valid': train_test['test']})
        tokenize_function: Callable = lambda examples: self.tokenizer(examples["text"])
        p: int = 1 # multiprocessing.cpu_count()
        tokenized_datasets: datasets.Dataset = dataset.map(tokenize_function, batched=True, num_proc=p, remove_columns=["text"])
        logging.info("Dataset tokenized!")

        self.dataset = tokenized_datasets.map(
            self._group_texts,
            batched=True,
            batch_size=1000,
            num_proc=p
        )
        n_train: int = len(self.dataset['train'])
        n_test: int = len(self.dataset['valid'])
        logging.info(f"Dataset prepared! Length of training ds: {n_train}. Length of test ds: {n_test}")
