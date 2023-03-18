UDL-Negation
==============================

Experiments to improve the weakness of misunderstanding the concept of negation in encoder-based models.

# Requirements

To install required libraries, run `pip install -r requirements.txt`.

# Create Data

To create the filterd training data, run the following four commands:

`make owt`
`make bc`
`make wiki`
`make cc_news`

Set paths and other configurations in `neg_udl/config/data_config.yaml`

WARNING: These operations might take several days!
WARNING: Undefined behavior experienced using multiprocessing!

# Run Experimeńts

Run experiments running the following commands (CUDA-capable device recommended):

```make exp_1_filtered```

```make exp_2_mlm+sup```

```make exp_3_mlm```
```make exp_3+_mlm```

Set paths and other configurations in `neg_udl/config/exp{1,2,3,3+}_config.yaml`

# Evaluate

To evaluate the trained model on selected GLUE-Tasks, run:

`make evaluate`

Project Organization
------------

    ├── LICENSE
    ├── Makefile                    <- Makefile with commands like `make data` or `make train`
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   ├── external                <- Data from third party sources.
    │   ├── interim                 <- Intermediate data that has been transformed.
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                   <- Jupyter notebooks.
    │
    ├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                 <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                       generated with `pip freeze > requirements.txt`
    │
    ├── evaluation
    │   └── figures                 <- Evaluation script for selected GLUE-tasks.
    │
    ├── setup.py                    <- makes project pip installable (pip install -e .) so src can be imported
    ├── neg_udl                     <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download and generate data
    │   │
    │   ├── Experiment.py  <-
    │   ├── MixedExperiment.py <-
    │   ├── 
    │   └── MLMNegMixed.py <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
