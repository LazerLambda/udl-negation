Code
==============================


------------

    ├── data                        <- Code for download and generating data.
    │
    ├── config                      <- Configuration files for data and experiments.
    │
    ├── Experiment.py               <- Base class for all experiments. Includes instantiation, training and
    │                                   evaluation (on test set and on oLMpics).
    │
    ├── MixedExperiment.py          <- Experiment-Child-Class for unsupervised (MLM) and supervised masking training.
    │                                   Includes a specific tokenization method to account for supervision.
    │
    ├── MLMExperiment.py            <- Experiment-Child-Class for MLM training (used on filtered data).
    │
    └── MLMNegMixed.py              <- Experiment-Child-Class for MLM training on filtered and WordNet data combined.

--------