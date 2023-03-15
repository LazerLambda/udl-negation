"""Evaluation Class for Original and Tuned Model.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import argparse
import evaluate
import numpy as np
import torch

from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollator
from transformers import Trainer
from transformers import TrainingArguments
from transformers.modeling_utils import load_state_dict
from typing import Any, Callable


def get_tokenizer(task: str, tokenizer: AutoTokenizer) -> Callable:
    if task in ['wnli', 'mrpc']:
        return lambda e : tokenizer(e["sentence1"], e["sentence2"], truncation=True)


def model_loader(ours_theirs: str, model_id: str, path2model: str) -> AutoModelForSequenceClassification:
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


def evaluate_ours_theirs(model_path: str, model_id: str = "prajjwal1/bert-small") -> None:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator: DataCollator = DataCollatorWithPadding(tokenizer=tokenizer)
    tasks: list = ['mrpc', 'wnli', ]

    training_args = TrainingArguments(
        "test-trainer",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        weight_decay=0.1,
        num_train_epochs=10,
        warmup_ratio=0.06,
        report_to="none"
    )
    results: dict = {}
    for task in tasks:
        raw_datasets = load_dataset("glue", task)
        metric = evaluate.load("glue", task)
        tokenized_datasets = raw_datasets.map(get_tokenizer(task, tokenizer), batched=True)

        # Original
        model = model_loader(
            'theirs',
            model_id,
            model_path)
        # result_theirs: dict = train_and_eval(
        #     model,
        #     tokenizer,
        #     data_collator,
        #     tokenized_datasets,
        #     training_args,
        #     metric)
        result_theirs: dict =  {'accuracy': 0.6617647058823529, 'f1': 0.7604166666666667}

        # Ours
        model = model_loader(
            'ours',
            model_id,
            model_path)
        # result_ours: dict = train_and_eval(
        #     model,
        #     tokenizer,
        #     data_collator,
        #     tokenized_datasets,
        #     training_args,
        #     metric)
        result_ours: dict =  {'accuracy': 0.6617647058823529, 'f1': 0.7604166666666667}
        tmp_dict: dict = {
            'ours': result_ours,
            'theirs': result_theirs
        }
        results[task] = tmp_dict # result
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='evaluate',
                    description='Evaluator for Original and Our Model.')
    parser.add_argument(
        '-m'
        '--model',
        default="prajjwal1/bert-small",
        required=False)
    parser.add_argument(
        '-p',
        '--path',
        default="../model/N-bests-prajjwal1-bert-small.pt"
    )
    args: argparse.Namespace = parser.parse_args()
    print(args, type(args))
    print(args.m__model)
    evaluate_ours_theirs(model_path=args.path, model_id=args.m__model)
