"""Unsupervised Deep-Learning Seminar.

Preprocess Book Corpus DS

LMU Munich
Philipp Koch, 2023
MIT-License
"""


import logging
import multiprocessing
import os
import pathlib
from multiprocessing import Pool
from typing import Callable, List, Tuple

import datasets
import hydra
import pandas as pd
import spacy
from datasets import load_dataset
from omegaconf import DictConfig


def process_bc(args: tuple, save_step: int = 2000) -> None:
    """Pre-Process Book Corpus Dataset.

    Function splits data from bc dataset into sentences and
    writes them on a line each to the target file. Newline characters
    are changed to space.

    :param args: Arguments for function in form of
        (process number, target path, (lower index bound, upper index bound))
    :param save_step: Step at which file is updated.
    """
    p_no, target_path, ind_range = args
    total: int = ind_range[1] - ind_range[0]
    dataset_bc = load_dataset("bookcorpus")
    nlp = spacy.load('en_core_web_sm')
    f = open(os.path.join(target_path, f"bc-{p_no}.txt"), "w")
    counter: int = 0
    sents: list = []
    neg: list = []
    for i in range(*ind_range):
        sent: str = dataset_bc['train'][i]['text']
        obj = nlp(sent)
        for token in obj:
            if token.dep_ == "neg":
                neg.append((counter + 1))
                break
        sents.append(str(sent).replace('\n', ' ') + '\n')
        counter += 1
        if i % save_step == 0 and i != 0:
            logging.info(f"Process {p_no}.\nWrite to file: {i - save_step} - {i}.\nTotal: {i - ind_range[0]}/{total}.")
            f.write(''.join(sents))
            sents = []
    f.write(''.join(sents))
    f.close()
    neg.append(f"n: {counter}")
    counter_dict: dict = {'neg': neg}
    pd.DataFrame(counter_dict).to_csv(os.path.join(target_path, f"bc-{p_no}.csv"))


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


@hydra.main(version_base=None, config_path="../config", config_name="data_config")
def main(cfg: DictConfig) -> None:
    """Run Main.

    :param cfg: Hydra config.
    """
    dataset_bc: datasets.Dataset = load_dataset("bookcorpus")
    n: int = len(dataset_bc['train'])
    n = int(cfg.preprocessing.bc.proportion * n)
    logging.info("Total length of new ds: ", str(n))
    p: int = multiprocessing.cpu_count()
    pathlib.Path(cfg.preprocessing.bc.target).mkdir(parents=True, exist_ok=True)
    with Pool(p) as pool:
        args: list = list(zip(
            range(1, n + 1),
            # ['bookcorpus'] * p,
            [cfg.preprocessing.bc.target] * p,
            get_ranges(n, p)))
        pool.map(process_bc, args)


if __name__ == '__main__':
    """Run Script."""
    main()
