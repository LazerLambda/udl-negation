"""Preprocess Wiki Dataset.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""


import datasets
import spacy
import multiprocessing
import numpy as np

from datasets import load_dataset
from multiprocessing import Pool
from typing import Callable, Tuple, Generator

def process_wiki(args: tuple, save_step: int = 1000) -> None:
    p_no, ind_range = args
    dataset_wikipedia = load_dataset("wikipedia", "20220301.en", beam_runner="DirectRunner")
    nlp = spacy.load('en_core_web_sm')
    f = open(f"wiki-{p_no}.txt", "a")
    counter = 0
    sents = []
    for i in range(*ind_range):
        obj = nlp(dataset_wikipedia['train'][i]['text'])
        for sent in obj.sents:
            check = False
            for token in sent:
                if token.dep_ == "neg":
                    check = True
            sents.append(str(sent.text).replace('\n', ' ') + '\n')
            counter += 1
        if i % save_step == 0 and i != 0:
            print(f"Write to file: {i} - {i - save_step}")
            f.write(''.join(sents))  
    f.write(''.join(sents))    
    f.close()

def get_ranges(n: int, p: int) -> Tuple[int, int]:
    ranges: Generator = zip(range(p ), [int(n / p)] * p)
    diff: int = int(n / p % 1 * 4)
    upper_bound: Callable = lambda e: (e[0] + 1) * e[1]
    return list(map(lambda e: (e[0] * e[1], upper_bound(e) if upper_bound(e) != n - diff else n), ranges))

def main() -> None:
    dataset_wikipedia: datasets.Dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        beam_runner="DirectRunner")
    n: int = len(dataset_wikipedia['train'])
    p: int = multiprocessing.cpu_count()
    with Pool(p) as pool:
        args: list = list(zip(range(1, n + 1), get_ranges(n, p)))
        pool.map(process_wiki, args)
    p.start()
    p.join()

if __name__ == '__main__':
    main()