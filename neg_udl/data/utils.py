"""Utils for Data-Preprocessing.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import os

import pandas as pd
import spacy
from datasets import load_dataset


def process(args: tuple, save_step: int = 2000) -> None:
    """Pre-Process Dataset.

    Function splits data into sentences and writes them on a line each to
    the target file. Newline characters are changed to space.

    :param args: Arguments for function in form of
        (process number, dataset string, target path,
         (lower index bound, upper index bound))
    :param save_step: Step at which file is updated.
    """
    p_no, ds, target_path, ind_range = args
    total: int = ind_range[1] - ind_range[0]
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        beam_runner="DirectRunner")\
            if ds == "wikipedia" else load_dataset(ds)
    nlp = spacy.load('en_core_web_sm')
    f = open(os.path.join(target_path, f"{ds}-{p_no}.txt"), "w")
    counter: int = 0
    sents: list = []
    neg: list = []
    for i in range(*ind_range):
        obj = nlp(dataset['train'][i]['text'])
        for sent in obj.sents:
            for token in sent:
                if token.dep_ == "neg":
                    neg.append(counter + 1)
            sents.append(str(sent.text).replace('\n', ' ') + '\n')
            counter += 1
        if i % save_step == 0 and i != 0:
            print(f"Process {p_no}.\nWrite to file: {i - save_step} - {i}.\nTotal: {i - ind_range[0]}/{total}.")
            f.write(''.join(sents))
            sents = []
    f.write(''.join(sents))
    f.close()
    neg.append(f"n: {counter}")
    counter_dict: dict = {'neg': neg}
    pd.DataFrame(counter_dict).to_csv(os.path.join(target_path, f"{ds}-{p_no}.csv"))