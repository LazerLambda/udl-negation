"""
Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import typing

import evaluate
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


class EvaluateGLUE():
    """ Class to Evaluate Model on GLUE."""

    def __init__(
            self,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.PreTrainedTokenizerBase,
            task: str,
            train_args: transformers.training_args.TrainingArguments
            ):
        """Iniatilize Class for Evaluation on GLUE.

        :param model: Model to be evaluated.
        :param tokenizer: Respective tokenizer for the model.
        :param task: String determining on which subtask of GLUE,
          the model is to be evaluated.
        :param train_args: Parameter for finetuning of the model.
        """
        self.model: transformers.PreTrainedModel = model
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer
        self.task: typing.Union[str, list] = task
        self.train_args: transformers.training_args.TrainingArguments =\
            train_args

        self.current_task: str = task
    
    def __tokenize(self, instance: str, task: str) -> typing.Any:
        """Tokenize Dataset for Model.
        
        :param instance: Sample from the respective dataset.
        :param task:
        """
        if task == 'mrpc':
            toret: typing.Any = tokenizer(
                instance["sentence1"],
                instance["sentence2"],
                truncation=True)
            print(type(toret))
            return toret
        else:
            raise NotImplementedError()
    
    def __load(self) -> None:
        raw_datasets = load_dataset("glue", self.task)
        self.tokenized_datasets = raw_datasets.map(self.__tokenize, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __train(self) -> None:
        self.trainer = Trainer(
            self.model,
            self.train_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        self.trainer.train()

    def evaluate(self) -> dict:
        self.__load()
        self.__train()
        predictions = self.trainer.predict(self.tokenized_datasets["validation"])
        preds = np.argmax(predictions.predictions, axis=-1)

        metric = evaluate.load("glue", self.task)
        return metric.compute(predictions=preds, references=predictions.label_ids)