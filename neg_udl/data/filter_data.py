"""Unsupervised Deep-Learning Seminar.

Main File to prepare Negation-Aware Data

LMU Munich
Philipp Koch, 2023
MIT-License
"""

import os
import pathlib
import re
from os import listdir
from os.path import isfile, join
from typing import IO, List, Match, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from DataServer import DataServer
from omegaconf import DictConfig


def make_data(path_target: str, path_data: str, path_csv: str, range_n: int, interval: int = 1000) -> None:
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
    :param interval: Interval, in which data will be saved.
    """
    df = pd.read_csv(path_csv, dtype={"neg": str})
    f: IO = open(path_data, 'r')
    n: int = range_n * 2 + 1
    indices: list = list(np.unique(df['neg'].values[:-1].astype(int)))
    next_neg: int = indices.pop(0)
    collector: DataServer = DataServer(path_target, n=n, interval=interval)
    for i, text in enumerate(f):
        collector.push(text)
        if i + 1 == int(next_neg) + range_n:
            collector.rolling_window()
            if len(indices) != 0:
                next_neg = indices.pop(0)

    begin: int = i + 1
    for j in range(begin, i + n):
        collector.push(None)
        if j + 1 == int(next_neg) + range_n:
            collector.rolling_window()
            if len(indices) != 0:
                next_neg = indices.pop(0)
    f.close()
    collector.done()


def regex_helper(regex: str, name: str) -> str:
    """Get Deterministic String from Regex-Match.

    :param regex: Regular expression.
    :param name: Name of element.
    :returns: Extracted match.
    """
    match: Optional[Match[str]] = re.search(regex, name)
    assert match is not None
    return match.group(1)


def get_paths(path: str) -> List[Tuple[str, str, str]]:
    """Get Paths for Data Collection.

    :param path: Path to folder in which output from 'make' scripts is outputted
        to.
    :returns: List of tuples containing paths to csv and txt files as well as the
        names of the respective files without ending.
    """
    files: list = [f for f in listdir(path) if isfile(join(path, f))]
    files_txt: list = list(filter(lambda e: bool(re.search(r"txt", e)), files))
    files_txt = list(map(lambda e: os.path.join(path, e), files_txt))
    file_names_cleaned: list = list(map(lambda e: regex_helper(r".*\/(.*).txt$", e), files_txt))
    files_csv: list = list(map(lambda e: os.path.join(path, e + '.csv'), file_names_cleaned))
    return list(zip(files_csv, files_txt, file_names_cleaned))


def combine_data(files: list, data_name: str) -> None:
    """Combine Data.

    :param files: Files that will be combined.
    :param data_name: Name of final data file.
    """
    head, _ = os.path.split(data_name)
    pathlib.Path(head).mkdir(parents=True, exist_ok=True)
    final_data: IO = open(data_name, 'w')
    for file in files:
        f_tmp: IO = open(file, 'r')
        final_data.write(f_tmp.read())
        f_tmp.close()
    final_data.close()


@hydra.main(version_base=None, config_path="../config", config_name="data_config")
def main(cfg: DictConfig):
    """Run Main.

    :param cfg: Hydra config.
    """
    tmp_names: list = []
    for key in cfg.preprocessing.tmp.keys_list:
        paths: List[Tuple[str, str, str]] = get_paths(cfg.preprocessing[key].target)
        for csv, txt, tail in paths:
            path_tmp: str = os.path.join(cfg.preprocessing.tmp.tmp_folder, tail + '_neg.txt')
            tmp_names.append(path_tmp)
            make_data(
                path_tmp,
                txt,
                csv,
                cfg.preprocessing.tmp.rolling_window)
    tmp_names.sort()
    combine_data(
        tmp_names,
        cfg.preprocessing.final.data_path)


if __name__ == "__main__":
    main()
