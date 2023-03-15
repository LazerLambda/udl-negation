"""Unsupervised Deep-Learning Seminar.

MLM-Based Experiment with Overrepresented Negation Data
and synthetic data.

LMU Munich
Philipp Koch, 2023
MIT-License
"""

import logging
import multiprocessing
import os
import pathlib
from typing import IO, Callable, Dict

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from .data.make_dataset import main
from .Experiment import Experiment


class MLMNegMixed(Experiment):
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

    def create_mixed_dataset(self, amount: int, path_orig: str, path_synth: str, path_target: str) -> None:
        """Create Mixed Dataset of Original and Synthetic Data.

        Count lines of original dataset and sample in the size of the synthetic dataset for equal
        representation in the data. Combine both datasets and shuffle dataset.

        :param amount: Amount of repetition of each line.
        :param path_orig: Path to original dataset.
        :param path_synth: Path to synthetic dataset.
        :param path_target: Path to target dataset.
        """
        f: IO = open(path_orig, 'r')
        counter: int = 0
        for _ in f:
            counter += 1
        f.close()
        indices: np.ndarray = np.random.choice(range(counter), amount, replace=False)
        indices.sort()
        indices = list(indices)
        f = open(path_orig, 'r')
        collector: list = []
        counter = 0
        for i, text in enumerate(f):
            if i == indices[counter]:
                collector.append(text)
                if counter >= len(indices) - 1:
                    break
                else:
                    counter += 1
        f.close()
        df_orig: pd.DataFrame = pd.DataFrame({'text': collector})
        df_synth: pd.DataFrame = pd.read_csv(path_synth, header=None)
        df_synth.columns = ['text']
        df_synth.text = list(map(lambda e: str(e) + '\n', df_synth.text.values))
        df: pd.DataFrame = pd.concat([df_orig, df_synth])
        df = df.sample(frac=1).reset_index(drop=True)

        head, _ = os.path.split(path_target)
        pathlib.Path(head).mkdir(parents=True, exist_ok=True)
        f = open(path_target, "w")
        f.writelines(map(lambda e: str(e), df.text.values))
        f.close()
        logging.info(f"Dataset written to {path_target}")

    def prepare_dataset(self) -> None:
        """Prepare Dataset for MLM Training.

        Set up dataset, load from .txt file, split, tokenize and
        eventually group data according to `blocksize` parameter.
        Requires 'path', 'test-prop' and 'blocksize' in dataset_config
        dict.
        """
        data_path: str = self.dataset_config['path']
        data_path_orig: str = self.dataset_config['path_orig']
        data_path_synth: str = self.dataset_config['path_synth']
        test_prop: float = self.dataset_config['test-prop']
        amount: int = self.dataset_config['amount']
        mp: bool = self.dataset_config['mp_activate']
        self.blocksize = self.dataset_config['blocksize']

        if not os.path.exists(data_path):  # Check if dataset exists
            logging.info(f"Dataset at {data_path} does not exist!")
            amount = main(data_path_synth, amount=amount, masked=False)
            self.create_mixed_dataset(amount, data_path_orig, data_path_synth, data_path)
        else:
            logging.info(f"Dataset at {data_path} exist!")

        data = load_dataset(
            "text", data_files=data_path, sample_by="line")
        train_test = data['train'].train_test_split(test_size=test_prop)
        dataset = datasets.DatasetDict({
            'train': train_test['train'],
            'valid': train_test['test']})
        tokenize_function: Callable = lambda examples: self.tokenizer(examples["text"])
        p: int = multiprocessing.cpu_count() if mp else 1
        tokenized_datasets: datasets.Dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=p,
            remove_columns=["text"])
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
