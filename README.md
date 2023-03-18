UDL-Negation
==============================

Code to the report "Comparing Data-Driven Techniques for Enhancing Negation Sensitivity in MLM-Based Laguage-Models".

Transformers in the field of language have shown impressive results in recent years. Despite the overall improvement, these models still lack an understanding of fundamental natural language concepts. One significant problem is the misunderstanding of the concept of negation. This work tested three different techniques to boost the model's understanding capabilities. These approaches include training a BERT-Small model on:
 - data with an increased amount of negations (filtered data) with MLM
 - filtered data (with MLM) and artificially generated data based on WordNet in an adversarial setting (WordNet adversarial data) by using supervised masking to guide the model in training
 - WordNet adversarial data and filtered data using only MLM.

# Requirements

To install required libraries, run `pip install -r requirements.txt`.

# Create Data

To create the filtered training data, run the following four commands:

`make owt`
`make bc`
`make wiki`
`make cc_news`

Set paths and other configurations in `neg_udl/config/data_config.yaml`

WARNING: These operations might take several days!
WARNING: Undefined behavior experienced when using multiprocessing!

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

# Project Organization
------------

    ├── LICENSE
    ├── Makefile                    <- Makefile with all necessary commands to generate data and run experiments.
    ├── data                        <- Create after cloning!
    │   ├── interim                 <- Intermediate data that has been transformed.
    │   └── processed               <- The final data used for experiments.
    │
    ├── models                      <- Create after cloning! Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks                   <- Jupyter notebooks (for plots)
    │
    ├── reports                     <- Report
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                       generated with `pip freeze > requirements.txt`
    │
    ├── evaluation                  <- Evaluation script for selected GLUE-tasks.
    │
    └── neg_udl                     <- Source code for use in this project.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
