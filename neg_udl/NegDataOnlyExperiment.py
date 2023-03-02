"""Negation Data Only Experiment.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import os
from typing import Any, Tuple

import datasets
import pandas as pd
import torch

from .data.make_dataset import main
from .Experiment import Experiment


class NegDataOnlyExperiment(Experiment):
    """Negation-Only Experiment.

    This experiment uses only the negation dataset
    (statement + word, statement + negated antonym) for
    pre-training tuning.
    """

    @staticmethod
    def equalize_data(
            unmasked: torch.Tensor,
            masked: torch.Tensor,
            mask_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add Mask Token to Shorter List.

        Sometimes it happens, that a masked word consists of two tokens (e.g.
        'foul ball' -> '<mask>'). To have equal length for masking, it is necessary,
        to have the same length. Thus, more masked tokens are added to to the
        masked sequence to account for the split (e.g. '<mask>' -> '<mask> <mask>').
        This is done in index space ([..., 50296, ...]-> [..., 50296, 50296, ...]).

        :param unmasked: Unmasked sequence (the longer sequence).
        :param masked: Masked sequence (the shorter sequence.)
        :param mask_token_id: Id of masked token from tokenizer.
        :returns: Equalized dataset.
        """
        ind: int = (masked == 50264).nonzero(as_tuple=True)[1].item()
        diff: int = unmasked.shape[1] - masked.shape[1] + 1
        return (
            torch.cat(
                (
                    masked[0][0:ind],
                    torch.tensor([mask_token_id] * diff),
                    masked[0][(ind + 1)::]
                ), 0).unsqueeze(0),
            torch.Tensor([1] * unmasked.shape[1]).to(torch.int8))

    def tokenize_dataset(self, elem: dict) -> dict:
        """Tokenize Dataset.

        Atomic function applied to each instance of the dataset.

        :param elem: Element of the dataset.
        :returns: Dictionary including 'input_ids', 'attention_mask' and
            'labels'.
        """
        attention_mask: list = []
        input_ids: list = []
        labels: list = []
        for elem_masked, elem_unmasked in zip(elem['x'], elem['y']):
            masked: torch.Tensor = self.tokenizer(elem_masked, return_tensors='pt')
            unmasked: torch.Tensor = self.tokenizer(elem_unmasked, return_tensors='pt')
            unm: torch.Tensor = unmasked['input_ids']
            msk: torch.Tensor = masked['input_ids']
            att: torch.Tensor = masked['attention_mask']
            if masked['input_ids'].shape != unmasked['input_ids'].shape:
                msk, att = self.equalize_data(unm, msk, self.tokenizer.mask_token_id)
            unm[(msk == unm)] = -100
            attention_mask.append(att.squeeze())
            input_ids.append(msk.squeeze().long().tolist())
            labels.append(unm.squeeze().long().tolist())
        result: dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return result

    def load_custom_dataset(
            self,
            path: str,
            test: float = 0.1,
            val: float = 0.5,
            sep="|") -> datasets.Dataset:
        """Load Custom Dataset for Experiment.

        :param path: Path to dataset (data.txt).
        :param test: Proportion of the test dataset.
        :param val: Proportion of the val dataset of the
            test dataset.
        :param sep: Seperator token for txt file (treated as .csv).
        :returns: Prepared dataset.
        """
        df: pd.DataFrame = pd.read_csv(path, sep=sep)
        data: Any = datasets.Dataset.from_pandas(df)

        train_test = data.train_test_split(test_size=test)
        test_valid = train_test['test'].train_test_split(test_size=val)
        dataset = datasets.DatasetDict({
            'train': train_test['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})

        return dataset.map(
            self.tokenize_dataset,
            batched=True,
            num_proc=4,
            remove_columns=["y", "x"])

    def prepare_dataset(self) -> None:
        """Prepare Dataset.

        Load custom dataset and assign to class variable.
        """
        data_path: str = self.dataset_config['path']
        test_prop: float = self.dataset_config['test-prop']
        val_prop: float = self.dataset_config['val-prop']

        if not os.path.exists(data_path):  # Check if dataset exists
            main(data_path)

        self.dataset = self.load_custom_dataset(data_path, test=test_prop, val=val_prop)
