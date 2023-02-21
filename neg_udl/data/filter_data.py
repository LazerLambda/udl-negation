"""Main File to prepare Negation-Aware Data.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import re
from os import listdir
from os.path import isfile, join
from typing import IO, List, Tuple

import os
import hydra
import pandas as pd
from DataServer import DataServer
from omegaconf import DictConfig


def make_data(path_target: str, path_data: str, path_csv: str, range_n: int) -> None:
    """Make Dataset.
    
    Iterate over dataset line-by-line and save data if line
    is mentioned in data-csv. To use rolling-window technique,
    the indices of specific data instances are increased by the
    length of the rolling window in one direction (`range_n`).
    If the condition is met, the whole rolling window is saved.
    To include further instances at the end of the file, another
    loop is used to account for the last instances.

    :param path_target: Path where final data will be stored.
    :param path_data: Path where raw data is stored.
    :param path_csv: Path to csv containing important rows and
        number of total raw-data instances in last row.
    :param range_n: Length of rolling-window in one direction
        (n_range + 1 + n_range).
    """
    df = pd.read_csv(path_csv)
    f: IO = open(path_data, 'r')
    n: int = range_n * 2 + 1
    indices: list = list(df['neg'].values[:-1].astype(int))
    next_neg: int = indices.pop(0)
    collector: DataServer = DataServer(path_target, n=n, interval=2)
    for i, text in enumerate(f):
        collector.push(text)
        if i + 1 == int(next_neg) + range_n:
            collector.rolling_window()
            if len(indices) != 0:
                next_neg = indices.pop(0)
            else:
                print("HIER", path_data)
                break

    begin: int = i + 1
    for j in range(begin, i + n):
        collector.push(None)
        if j + 1 == int(next_neg) + range_n:
            collector.rolling_window()
            if len(indices) != 0:
                next_neg = indices.pop(0)
            else:
                break
    f.close()

def get_paths(path: str) -> List[Tuple[str, str, str]]:
    files: list = [f for f in listdir(path) if isfile(join(path, f))]
    files_txt: list = list(filter(lambda e: bool(re.search(r"txt", e)), files))
    files_txt = list(map(lambda e: os.path.join(path, e), files_txt))
    file_names_cleaned: list = list(map(lambda e: re.match(r".*\/(.*).txt$", e).group(1), files_txt))
    files_csv: list = list(map(lambda e: os.path.join(path, e + '.csv'), file_names_cleaned))
    return list(zip(files_csv, files_txt, file_names_cleaned))

def combine_data(files: list, data_name: str) -> None:
    f: IO = open(data_name, 'w')
    for file in files:
        f_tmp: IO = open(file, 'r')


@hydra.main(version_base=None, config_path="../config", config_name="data_config")
def main(cfg: DictConfig):
    print(cfg['preprocessing'])
    paths = get_paths(cfg.preprocessing.bc.target)
    tmp_names: list = []
    for csv, txt, tail in paths:
        path_tmp: str = os.path.join(cfg.preprocessing.tmp.tmp_folder, tail + '_neg.txt')
        tmp_names.append(path_tmp)
        make_data(
            path_tmp,
            txt,
            csv,
            cfg.preprocessing.tmp.rolling_window)


if __name__ == "__main__":
    main()