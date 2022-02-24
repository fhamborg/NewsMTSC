# NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles
NewsMTSC is a dataset for target-dependent sentiment classification (TSC) on
news articles reporting on policy issues. The dataset consists of more than 11k labeled
sentences, which we sampled from news articles from online US news outlets. More
information can be found in our paper published at the EACL 2021.

This repository contains the **dataset** for target-dependent
sentiment classification in news articles reporting on policy issues. Additionally,
the repository contains our **model** named GRU-TSC, which achieves state-of-the-art
TSC classification performance on NewsMTSC. Check it out - it **works out of the box** :-)

**Quick start**

* **I want to classify sentiment**: check out our easy-to-use, high-quality sentiment classifier on [PyPI](https://pypi.org/project/NewsSentiment/)
* **I need the dataset**: you can [download it here](https://github.com/fhamborg/NewsMTSC/raw/main/NewsSentiment/controller_data/datasets/NewsMTSC-dataset/NewsMTSC-dataset.zip) or [view it here](https://github.com/fhamborg/NewsMTSC/tree/main/NewsSentiment/controller_data/datasets/NewsMTSC-dataset). We also offer NewsMTSC as a dataset on [Huggingface Hub](https://huggingface.co/datasets/fhamborg/news_sentiment_newsmtsc) and on [Kaggle](https://www.kaggle.com/fhamborg/news-articles-sentiment).
* **I want to train my own models**: read the remainder of this file.


# Installation
It's super easy, we promise! Note that following these instructions is only necessary if you're planning to train a model using our tool. If you only want to predict the sentiment of sentences, please use our [Python package](https://pypi.org/project/NewsSentiment/), which is even easier to install and use :-)

NewsMTSC was tested on MacOS and Ubuntu; other OS may work, too. Let us know :-)

**1. Setup the environment:**

This step is optional if you have Python 3.7 installed already (`python --version`). If you don't have Python 3.7, we recommend using Anaconda for setting up requirements. If you do not have it yet, follow Anaconda's
[installation instructions](https://docs.anaconda.com/anaconda/install/). 

To setup a Python 3.7 environment (in case you don't have one yet) you may use, for example:
```bash
conda create --yes -n newsmtsc python=3.7
conda activate newsmtsc
```

FYI, for users of virtualenv, the equivalent command would be:
```bash
virtualenv -ppython3.7 --setuptools 45 venv
source venv/bin/activate
```

**2. Setup NewsMTSC:**
```bash
git clone git@github.com:fhamborg/NewsMTSC.git
```

Afterward, for example, open the project in your IDE and follow the instruction described in the section "Training".

Note that if you only want to classify sentiment using our model, we recommend that you use our PyPI package [NewsSentiment](https://pypi.org/project/NewsSentiment/). Getting it is as simple as `pip install NewsSentiment` and using it is four lines of code :-)


# Training
If you want to train one of our models or your own model, please clone the repository first.


There are two entry points to the system. `train.py` is used to train and evaluate a specific model on a specific dataset using
specific hyperparameters. We call a single run an _experiment_. `controller.py` is used to run multiple experiments
automatically. This is for example useful for model selection and evaluating hundreds or thousands of combinations of
models, hyperparameters, and datasets.

## Running a single experiment
Goal: training a model with a user-defined (hyper)parameter combination.

`train.py` allows fine-grained control over the training and evaluation process, yet for most command line arguments
we provide useful defaults. Two arguments are required:

* `--own_model_name` (which model is used, e.g., `grutsc`),
* `--dataset_name` (which dataset is used, e.g., `newsmtsc-rw`).

For more information refer to `train.py` and
`combinations_absadata_0.py`. If you just want to get started quickly, the command below should work out of the box. 

```
python train.py --own_model_name grutsc --dataset_name newsmtsc-rw
```

## Running multiple experiments 
Goal: finding the (hyper)parameter combination to train a model that achieves the best performance.

`controller.py` takes a set of values for each argument, creates combinations of arguments, applies conditions to remove
unnecessary combinations (e.g., some arguments may only be used for a specific model), and creates a multiprocessing
pool to run experiments of these argument combinations in parallel. After completion, `controller.py` creates a summary,
which contains detailed results, including evaluation performance, of all experiments. By using `createoverview.py`, you
can export this summary into an Excel spreadsheet.   

# Acknowledgements
This repository is in part based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).
We thank Song et al. for making their excellent repository open source.

# How to cite
If you use the dataset or model, please cite our [paper](https://www.aclweb.org/anthology/2021.eacl-main.142/) ([PDF](https://www.aclweb.org/anthology/2021.eacl-main.142.pdf)):

```
@InProceedings{Hamborg2021b,
  author    = {Hamborg, Felix and Donnay, Karsten},
  title     = {NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)},
  year      = {2021},
  month     = {Apr.},
  location  = {Virtual Event},
}
```
