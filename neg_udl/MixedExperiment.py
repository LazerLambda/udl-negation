"""
Mixed Data Experiment.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import os
from typing import Any, IO

import logging
import datasets
import pandas as pd
import pathlib
import numpy as np
import torch

from .data.make_dataset import main
from .Experiment import Experiment


class MixedExperiment(Experiment):
    """Mixed Experiment.

    This experiment uses only the negation dataset
    (statement + word, statement + negated antonym) for
    pre-training tuning.
    """

    def process_dataset(self, inputs: dict, mlm_probabilty: float =0.15):
        """"""
        tokenized: dict = self.tokenizer(inputs['text'], return_tensors='pt', max_length=self.max_length, truncation=True)

        if (self.end_index in tokenized['input_ids']) and (self.begin_index in tokenized['input_ids']):
            tokenized = {k : v.squeeze() for k, v in tokenized.items()}

            # Find masked word in reference sequence
            bool_vector: torch.Tensor = torch.ones(len(tokenized['input_ids']), dtype=torch.bool)
            begin: int = ((tokenized['input_ids'].squeeze() == self.begin_index).nonzero(as_tuple=True)[0])[0].item()
            end: int = ((tokenized['input_ids'].squeeze() == self.end_index).nonzero(as_tuple=True)[0])[0].item()
            rm: torch.Tensor = torch.zeros(end + 1 - begin, dtype=torch.bool)
            bool_vector[torch.Tensor(range(begin, end + 1)).long()] = rm

            # Get sentence (masked) and label (known word)
            orig: torch.Tensor = torch.masked_select(tokenized['input_ids'], bool_vector)
            label: torch.Tensor = torch.masked_select(tokenized['input_ids'], ~bool_vector)
            # Remove special tokens around label
            label = label[1::]
            label = label[:-1]

            # Get split sequences 
            ind: int = (orig == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
            label_f: tuple = (
                torch.zeros(ind, dtype=torch.long) - 100, label,
                torch.zeros(len(orig[(ind + 1):: ]), dtype=torch.long) - 100)
            elem: tuple = (orig[0:ind], torch.tensor([self.tokenizer.mask_token_id] * len(label)), orig[(ind + 1):: ])

            # Concatenate splitted sequences and prepare for return
            elem: torch.Tensor = torch.concat(elem)
            label_f: torch.Tensor = torch.concat(label_f)
            attention_mask: torch.Tensor = torch.ones(elem.shape, dtype=torch.long)
            
            # Apply padding
            elem = torch.cat((elem, torch.Tensor([self.tokenizer.pad_token_type_id] * (self.max_length - len(elem))).long()))
            label_f = torch.cat((label_f, (torch.zeros(self.max_length - len(label_f), dtype=torch.long) - 100)))
            attention_mask = torch.cat((attention_mask, torch.zeros(self.max_length - len(attention_mask), dtype=torch.long)))
            return {
                'input_ids': elem,
                'labels': label_f,
                'attention_mask': attention_mask}
        else:
            # MLM-Masking, according to Devlin et al. 2018.
            input_ids = tokenized['input_ids']
            labels = input_ids.clone()
            # Sample mlm_probability% of all appropriate tokens.
            probability_matrix = torch.full(labels.shape, mlm_probabilty)
            special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100

            # 80 % will be masked
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 15 % will be randomly filled.
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]
            return {
                'input_ids': torch.cat((input_ids.squeeze(), self.tokenizer.pad_token_type_id * torch.ones((self.max_length - input_ids.shape[1]), dtype=torch.long))),
                'labels': torch.cat((labels.squeeze(), torch.zeros(self.max_length - labels.shape[1], dtype=torch.long))),
                'attention_mask': torch.cat((torch.ones(input_ids.shape[1], dtype=torch.long), torch.zeros(self.max_length - input_ids.shape[1], dtype=torch.long)))
            }
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones(input_ids.shape[1], dtype=torch.long).unsqueeze(0)}

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
        # Add special tokens for data preparation
        self.tokenizer.add_tokens(["[REF-BEG]", "[REF-END]"])
        self.begin_index: int = self.tokenizer.vocab_size
        self.end_index: int = self.tokenizer.vocab_size + 1

        self.max_length: int = self.tokenizer.model_max_length
        data: datasets.Dataset = datasets.load_dataset(
            "text", data_files=path, sample_by="line")

        train_test = data['train'].train_test_split(test_size=test)

        dataset = datasets.DatasetDict({
            'train': train_test['train'].map(
                self.process_dataset,
                new_fingerprint='2130897980712098357',
                remove_columns=['text']
            ),
            'valid': train_test['test'].map(
                self.process_dataset,
                new_fingerprint='9286897880712091234',
                remove_columns=['text']
            )})
        n_train: int = len(dataset['train'])
        n_test: int = len(dataset['valid'])
        logging.info(f"Dataset prepared! Length of training ds: {n_train}. Length of test ds: {n_test}")

        return dataset
    
    def create_mixed_dataset(self, amount: int, path_orig: str, path_synth: str, path_target: str) -> None:
        """"""
        f:IO = open(path_orig, 'r')
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
        """Prepare Dataset.

        Load custom dataset and assign to class variable.
        """
        data_path: str = self.dataset_config['path']
        data_path_orig: str = self.dataset_config['path_orig']
        data_path_synth: str = self.dataset_config['path_synth']
        test_prop: float = self.dataset_config['test-prop']
        val_prop: float = self.dataset_config['val-prop']

        if not os.path.exists(data_path):  # Check if dataset exists
            logging.info(f"Dataset at {data_path} does not exist!")
            amount: int = main(data_path_synth)
            self.create_mixed_dataset(amount, data_path_orig, data_path_synth, data_path)
        else:
            logging.info(f"Dataset at {data_path} exist!")


        self.dataset = self.load_custom_dataset(data_path, test=test_prop, val=val_prop)
