# NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles
NewsMTSC is a dataset for target-dependent sentiment classification (TSC) on
news articles reporting on policy issues. The dataset consists of more than 11k labeled
sentences, which we sampled from news articles from online US news outlets. More
information can be found in our paper published at the EACL 2021.

This repository contains the dataset for target-dependent
sentiment classification in news articles reporting on policy issues. Additionally,
the repository contains our model named GRU-TSC, which achieves state-of-the-art
TSC classification performance on NewsMTSC. Check it out - it works out of the box :-)

This readme consists of the following parts:
* [Installation of GRU-TSC](#installation)
* [How to use our model](#target-dependent-sentiment-classification)
* [How to train our model (or yours)](#training)

If you are only looking for the dataset, you can [download it here](https://github.com/fhamborg/NewsMTSC/raw/main/controller_data/datasets/NewsMTSC-dataset/NewsMTSC-dataset.zip)
or [view it here](https://github.com/fhamborg/NewsMTSC/tree/master/controller_data/datasets/NewsMTSC-dataset).

To make available the model also to non-experts in computer science,
we aimed to make the installation and using the model as easy as possible. If you face
any issue with using the model or notice an issue in our dataset, please feel free to
open an issue.

# Installation
It's super easy, we promise!

To keep things easy, we use Anaconda for setting up requirements. If you do not have
it yet, follow Anaconda's
[installation instructions](https://docs.anaconda.com/anaconda/install/).
NewsMTSC was tested on MacOS and Ubuntu; other OS may work, too. Let us know :-)

**1. Setup the conda environment:**
```bash
conda create --yes -n newsmtsc python=3.7
conda activate newsmtsc
```

**2. Clone the repository:**
```bash
git clone git@github.com:fhamborg/NewsMTSC.git
cd NewsMTSC
```

**3. Install pytorch:**

Choose either of the following. Either use this command if your GPU supports CUDA:
```bash
conda install --yes "pytorch=1.7.1" torchvision cudatoolkit=10.1 -c pytorch
```

Or use this command if your GPU does not support CUDA, you don't know what CUDA is, or if the previous command gives you an error:
```bash
conda install --yes "pytorch=1.7.1" torchvision -c pytorch
```

**4. Install remaining packages:**
```bash
conda install --yes pandas tqdm scikit-learn
conda install --yes -c conda-forge boto3 regex sacremoses jsonlines matplotlib tabulate imbalanced-learn "spacy>=2.1,<3"
conda install --yes -c anaconda requests gensim openpyxl networkx
pip install "transformers>=3.1.0,<4"
python -m spacy download en_core_web_sm
```

**5. Download our model:**
```bash
python download.py
```

You're all set now :-)

# Target-dependent Sentiment Classification
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
