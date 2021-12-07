# NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles
NewsMTSC is a dataset for target-dependent sentiment classification (TSC) on
news articles reporting on policy issues. The dataset consists of more than 11k labeled
sentences, which we sampled from news articles from online US news outlets. More
information can be found in our paper published at the EACL 2021.

This repository contains the **dataset** for target-dependent
sentiment classification in news articles reporting on policy issues. Additionally,
the repository contains our **model** named GRU-TSC, which achieves state-of-the-art
TSC classification performance on NewsMTSC. Check it out - it **works out of the box** :-)

This readme consists of the following parts:
* [Installation of GRU-TSC](#installation)
* [How to use our model](#target-dependent-sentiment-classification)
* [How to train our model (or yours)](#training)

If you are only looking for the dataset, you can [download it here](https://github.com/fhamborg/NewsMTSC/raw/main/controller_data/datasets/NewsMTSC-dataset/NewsMTSC-dataset.zip)
or [view it here](https://github.com/fhamborg/NewsMTSC/tree/master/controller_data/datasets/NewsMTSC-dataset).

To make the model available also to users without programming skills,
we aimed to make the installation and using the model as easy as possible. If you face
any issue with using the model or notice an issue in our dataset, you are more than welcome to [open
an issue](https://github.com/fhamborg/NewsMTSC/issues).

# Installation
It's super easy, we promise!

To keep things easy, we use Anaconda for setting up requirements. If you do not have
it yet, follow Anaconda's
[installation instructions](https://docs.anaconda.com/anaconda/install/).
NewsMTSC was tested on MacOS and Ubuntu; other OS may work, too. Let us know :-)

We currently still require python 3.8. If your package manager offers it, just install it from there. Otherwise, you can
install it e.g. via coda:

**1. Setup the environment:**

Either via virtualenv:
```bash
virtualenv -ppython3.7 --setuptools 45 venv
source venv/bin/activate
```
or via conda:
```bash
conda create --yes -n newsmtsc python=3.7
conda activate newsmtsc
```

**2. Install NewsSentiment:**
```bash
pip install NewsSentiment        # without cuda support
pip install NewsSentiment[cuda]  # with cuda support
```

You're all set now, all required models will automatically download on-demand :-)

# Target-dependent Sentiment Classification

_Please note that running infer.py (or its first import) and the first run of TargetSentimentClassifier can take some time depending on your internet connection speed._
_NewsSentiment downloads and loads the required models during this time._

Target-dependent sentiment classification works out-of-the-box. Have a look at infer.py or give it a try:
```
python infer.py
```

# Training
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
