"""Evaluation Class for Original and Tuned Model.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import argparse
import json
import logging
from typing import Any, Callable

import evaluate
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollator, DataCollatorWithPadding, Trainer,
                          TrainingArguments)
from transformers.modeling_utils import load_state_dict


def get_tokenizer(task: str, tokenizer: AutoTokenizer) -> Callable:
    """Get Tokenizer-Function Depending on Subtask.

    :param task: String identifying subtask (e.g. 'mnli', 'sst2', ...)
    :param tokenizer: Tokenizer to be used in returned function.
    :returns: Function taking one argument for tokenizing respective input
    :raises: Exception if task is misspecified.
        depending on task.
    """
    if task in ['wnli', 'mrpc', 'rte', 'stsb']:
        return lambda e: tokenizer(e["sentence1"], e["sentence2"], truncation=True, max_length=512)
    if task in ['mnli_mismatched', 'mnli', 'mnli_matched']:
        return lambda e: tokenizer(e["premise"], e["hypothesis"], truncation=True, max_length=512)
    if task in ['sst2', 'cola']:
        return lambda e: tokenizer(e["sentence"], truncation=True, max_length=512)
    if task == 'qqp':
        return lambda e: tokenizer(e["question1"], e["question2"], truncation=True, max_length=512)
    if task == 'qnli':
        return lambda e: tokenizer(e["question"], e["sentence"], truncation=True, max_length=512)
    else:
        raise Exception(f"Wrong task! Obtained argument: '{task}'.")


def model_loader(ours_theirs: str, model_id: str, path2model: str) -> AutoModelForSequenceClassification:
    """Load Model Depending on Identifier.

    :param ours_theirs: Either 'ours' (tuned) or 'theirs' (original).
    :param model_id: Model identifier string (transformers library).
    :param path2model: Path to saved .pt model.
    :returns: Requested model.
    :raises: Exception, if unmatching `ours_theirs` is specified.
    """
    if ours_theirs == "ours":
        return AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            state_dict=load_state_dict(path2model))
    if ours_theirs == "theirs":
        return AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2)
    else:
        raise Exception(f"Argument `ours_theirs` must be either 'ours' or 'theirs'. Current is '{str(ours_theirs)}'")


def train_and_eval(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        data_collator: DataCollator,
        tokenized_datasets: Dataset,
        training_args: TrainingArguments,
        metric: Any) -> dict:
    """Train and Evaluate Model.

    :param model: Model to be fine-tuned (trained) and evaluated.
    :param tokenizer: Tokenizer, required for trainer.
    :param data_collator: DataCollator for trainer.
    :param tokenized_datasets: Already tokenized dataset for trainer.
    :param training_args: Training args for trainer.
    :param metric: Metric for evaluation (from evaluate class, specified for 'GLUE').
    :returns: Dict including keys for metrics (e.g. accuracy, f1, matthew's corr. coeff. etc.)
    """
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    predictions: torch.Tensor = trainer.predict(tokenized_datasets["validation"])
    preds: np.ndarray = np.argmax(predictions.predictions, axis=-1)
    return metric.compute(predictions=preds, references=predictions.label_ids)


def main(model_path: str, result_path: str, model_id: str = "prajjwal1/bert-small") -> dict:
    """Run Program.

    :param model_path: Model path to .pt file.
    :param result_path: Path to results .json file.s
    :param model_id: Model identifier string (transformers library).
    :returns: Dictionary with results (keys for tasks with another dictionary for
        original and tuned model.)
    """
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator: DataCollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=512)
    tasks: list = ['wnli', 'cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']

    training_args = TrainingArguments(
        "test-trainer",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        weight_decay=0.1,
        num_train_epochs=10,
        warmup_ratio=0.06,
        report_to="none",
        save_strategy='no'
    )
    results: dict = {}
    n: int = len(tasks)
    for i, task in enumerate(tasks):
        logging.info(f"Starting with task: {task}. {i}/{n}.")
        raw_datasets = load_dataset("glue", task)
        metric = evaluate.load("glue", task)
        tokenized_datasets = raw_datasets.map(get_tokenizer(task, tokenizer), batched=True)

        # Original
        result_theirs: dict = train_and_eval(
            model_loader(
                'theirs',
                model_id,
                model_path),
            tokenizer,
            data_collator,
            tokenized_datasets,
            training_args,
            metric)

        # Ours
        result_ours: dict = train_and_eval(
            model_loader(
                'ours',
                model_id,
                model_path),
            tokenizer,
            data_collator,
            tokenized_datasets,
            training_args,
            metric)
        tmp_dict: dict = {
            'ours': result_ours,
            'theirs': result_theirs
        }
        results[task] = tmp_dict
        json.dump(results, open(f"{result_path}.json", "w"))
        logging.info(f"Results saved to '{result_path}.json'!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='evaluate',
        description='Evaluator for Original and Our Model.')
    parser.add_argument(
        '-m'
        '--model',
        default="prajjwal1/bert-small",
        type=str)
    parser.add_argument(
        '-p',
        '--path',
        default="./models/N-prajjwal1-bert-small.pt",
        type=str
    )
    parser.add_argument(
        '-v', '--verbose',
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    parser.add_argument(
        '-r',
        '--result-path',
        default="GLUEResults",
        type=str,
    )
    args: argparse.Namespace = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    logging.info(f"Args provided:\n\t-path: '{args.path}'\n\t-model: '{args.m__model}'\n\t-result-path: '{args.result_path}'\n\t-Log-Level: INFO")
    results: dict = main(model_path=args.path, result_path=args.result_path, model_id=args.m__model)
    json.dump(results, open(f"{args.result_path}.json", "w"))
    logging.info(f"Results saved to '{args.result_path}.json'!")
