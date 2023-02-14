"""Preprocess Wiki Dataset.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""


import multiprocessing
import os
import pathlib
from multiprocessing import Pool
from typing import Callable, List, Tuple

import datasets
import pandas as pd
import spacy
from datasets import load_dataset

TARGET_PATH: str = './data/interim/wiki/'


def process_wiki(args: tuple, save_step: int = 1000) -> None:
    """Pre-Process Wikipedia Dataset.

    Function splits data from wikipedia dataset into sentences and
    writes them on a line each to the target file. Newline characters
    are changed to space.

    :param args: Arguments for function in form of
        (process number, (lower index bound, upper index bound))
    :param save_step: Step at which file is updated.
    """
    p_no, ind_range = args
    dataset_wikipedia = load_dataset("wikipedia", "20220301.en", beam_runner="DirectRunner")
    nlp = spacy.load('en_core_web_sm')
    f = open(os.path.join(TARGET_PATH, f"wiki-{p_no}.txt"), "w")
    counter: int = 0
    sents: list = []
    neg: list = []
    for i in range(*ind_range):
        obj = nlp(dataset_wikipedia['train'][i]['text'])
        for sent in obj.sents:
            for token in sent:
                if token.dep_ == "neg":
                    neg.append(counter + 1)
            sents.append(str(sent.text).replace('\n', ' ') + '\n')
            counter += 1
        if i % save_step == 0 and i != 0:
            print(f"Write to file: {i - save_step} - {i}.")
            f.write(''.join(sents))
    f.write(''.join(sents))
    f.close()
    neg.append(f"n: {counter}")
    counter_dict: dict = {'neg': neg}
    pd.DataFrame(counter_dict).to_csv(os.path.join(TARGET_PATH, f"wiki-{p_no}.csv"))


def get_ranges(n: int, p: int, lower: int = 0) -> List[Tuple[int, int]]:
    """Create Ranges List.

    Creates a list of ranges from `lower` to `n` in `p` parts.
    E.g.: 16, 4, 0: [(0,3), (4,7), (8,11), (12,15)].

    :param n: Upper bound of indices.
    :param p: Number of processes (length of final list).
    :param lower: Lower bound of list.
    :returns: List of tuples of ranges.
    """
    ranges: zip[Tuple[int, int]] = zip(range(lower, p), [int(n / p)] * p)
    diff: int = int(n / p % 1 * 4)
    upper_bound: Callable = lambda e: (e[0] + 1) * e[1]
    return list(map(lambda e: (e[0] * e[1], upper_bound(e) if upper_bound(e) != n - diff else n), ranges))


if __name__ == '__main__':
    """Run Script."""
    dataset_wikipedia: datasets.Dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        beam_runner="DirectRunner")
    n: int = len(dataset_wikipedia['train'])
    n = int(0.0001 * n)
    print("Total length of new ds: ", str(n))
    p: int = multiprocessing.cpu_count()
    pathlib.Path(TARGET_PATH).mkdir(parents=True, exist_ok=True)
    with Pool(p) as pool:
        args: list = list(zip(range(1, n + 1), get_ranges(n, p)))
        pool.map(process_wiki, args)
