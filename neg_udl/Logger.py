"""
Unsupervised Deep-Learning Seminar
LMU Munich
Philipp Koch, 2023
MIT-License
"""

import wandb

class Logger:
    """Logger Class."""

    def __init__(self, logger: str):
        if logger == "wandb":
            pass
        else:
            raise Exception("")