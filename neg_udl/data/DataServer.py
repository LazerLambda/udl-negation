"""Class to handle Rolling Window.

Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import os
import pathlib
from typing import IO, Any


class DataServer():
    """Class for Storing and Writing Data.

    This class stores values that are pushed to the internal list
    over a specific length for a rolling window. If a condition is met,
    the data from the rolling-window can be saved to an internal collection
    that is written to a file periodically (use interval parameter to define
    length).
    """

    def __init__(self, target_path: str, n: int = 5, interval: int = 1000):
        """Instantiate Class.

        :param target_path: Path to where file with data will be saved.
        :param n: Max length of internal list for rolling-window.
        :param interval: Length of interval in which the internal collection
            of rolling-windows is saved.
        """
        self.list: list = []
        self.collection: list = []
        self.target_path: str = target_path
        head, _ = os.path.split(target_path)
        pathlib.Path(head).mkdir(parents=True, exist_ok=True)
        self.n: int = n
        self.interval: int = interval

    def push(self, elem: Any) -> None:
        """Push to Rolling-Window List.

        Append element to rolling window list and pop first element
        of list iff max-length is reached.
        :param elem: Element to be appended to rolling window list.
        """
        if elem is None:
            self.list.pop(0)
            return
        if len(self.list) >= self.n:
            self.list.pop(0)
            self.list.append(elem)
        else:
            self.list.append(elem)

    def rolling_window(self) -> str:
        """Get and Save Rolling-Window List.

        :returns: String of joined content of rolling-window list with appended
            new-line-character.
        """
        data: str = " ".join(list(map(lambda e: str(e)[:-1] if str(e)[-1] == '\n' else str(e), self.list))) + "\n"
        self.collection.append(data)
        self._check_collection()
        return data

    def _check_collection(self) -> None:
        """Check and Save Collection.

        If length of the interval is reached, the content of the internal
        collection is written to file.
        """
        if len(self.collection) >= self.interval:
            f: IO = open(self.target_path, "a")
            f.writelines(line for line in self.collection)
            f.close()
            self.collection = []
            pass
