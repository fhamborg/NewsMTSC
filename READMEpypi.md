# NewsSentiment: easy-to-use, high-quality target-dependent sentiment classification for news articles
NewsSentiment is an easy-to-use Python library that achieves state-of-the-art performance
for target-dependent sentiment classification on news articles.
NewsSentiment uses the currently [best performing](https://aclanthology.org/2021.eacl-main.142.pdf) 
targeted sentiment classifier for news articles. In contrast to regular sentiment
classification, targeted sentiment classification allows you to provide a target in a sentence. 
Only for this target, the sentiment is then predicted. This is more reliable in many
cases, as demonstrated by the following simplistic example: "I like Bert, but I hate Robert."

We designed NewsSentiment to serve as an easy-to-use wrapper around the sophisticated
GRU-TSC model, which was trained on the NEWSMTSC dataset consisting of more than 10k 
labeled sentences sampled from political news articles. More information on the dataset 
and the model can be found [here](https://aclanthology.org/2021.eacl-main.142.pdf). The
dataset, the model, and its source code can be viewed in our [GitHub repository](https://github.com/fhamborg/NewsMTSC).

# Installation
It's super easy, we promise!

NewsMTSC was tested on MacOS and Ubuntu; other OS may work, too. Let us know :-)

**1. Setup the environment:**

This step is optional if you have Python 3.7 installed already (run `python --version` 
in a terminal and check the version that is printed). If you don't have Python 3.7, we 
recommend using Anaconda for setting up requirements because it is very easy (but any way
of installing Python 3.7 is fine). If you do not have Anaconda yet, follow their
[installation instructions](https://docs.anaconda.com/anaconda/install/). 

After installing Anaconda, to setup a Python 3.7 environment (in case you don't have one
yet) execute:
```bash
conda create --yes -n newsmtsc python=3.7
conda activate newsmtsc
```

**2. Install NewsSentiment:**
```bash
pip3 install NewsSentiment        # without cuda support (choose this if you don't know what cuda is)
pip3 install NewsSentiment[cuda]  # with cuda support
```

You're all set now, all required models will automatically download on-demand :-) 

Note that using NewsSentiment the first time may take a few minutes before the model is fully
download. This is a one-time process and future use of NewsSentiment will be much faster.

# Target-dependent Sentiment Classification

```python
from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

sentiment = tsc.infer_from_text("I like" ,"Peter", ".")
print(sentiment)
```

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
