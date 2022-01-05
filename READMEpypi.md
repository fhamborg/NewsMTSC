# NewsSentiment: easy-to-use, high-quality target-dependent sentiment classification for news articles
NewsSentiment is an easy-to-use Python library that achieves state-of-the-art performance
for target-dependent sentiment classification on news articles.
NewsSentiment uses the currently [best performing](https://aclanthology.org/2021.eacl-main.142.pdf) 
targeted sentiment classifier for news articles. In contrast to regular sentiment
classification, targeted sentiment classification allows you to provide a target in a sentence. 
Only for this target, the sentiment is then predicted. This is more reliable in many
cases, as demonstrated by the following simplistic example: "I like Bert, but I hate Robert."

We designed NewsSentiment to serve as an easy-to-use wrapper around the sophisticated
GRU-TSC model, which was trained on the NewsMTSC dataset consisting of more than 10k 
labeled sentences sampled from political news articles. More information on the dataset 
and the model can be found [here](https://aclanthology.org/2021.eacl-main.142.pdf). The
dataset, the model, and its source code can be viewed in our [GitHub repository](https://github.com/fhamborg/NewsMTSC).

# Installation
It's super easy, we promise! 

You just need a Python 3.7 or Python 3.8 environment. See [here](https://raw.githubusercontent.com/fhamborg/NewsMTSC/main/pythoninfo.md) if you 
don't have Python or a different version (run `python --version` in a terminal to see 
your version). Then run:

```bash
pip3 install NewsSentiment        # without cuda support (choose this if you don't know what cuda is)
pip3 install NewsSentiment[cuda]  # with cuda support
```

You're all set now :-)

# Target-dependent Sentiment Classification

Note that using NewsSentiment the first time will take *a few minutes* because it needs
to download the fine-tuned language model. Please do not abort this initial download. 
Since this is a one-time process, future use of NewsSentiment will be much faster.

```python
from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

sentiment = tsc.infer_from_text("I like " ,"Peter", " but I don't like Robert.")
print(sentiment[0])

sentiment = tsc.infer_from_text("" ,"Mark Meadows", "'s coverup of Trump’s coup attempt is falling apart.")
print(sentiment[0])
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
