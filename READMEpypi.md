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

You just need a Python 3.8 environment. See [here](https://raw.githubusercontent.com/fhamborg/NewsMTSC/main/pythoninfo.md) if you 
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

data = [
    ("I like ", "Peter", " but I don't like Robert."),
    ("", "Mark Meadows", "'s coverup of Trump’s coup attempt is falling apart."),
]

sentiments = tsc.infer(targets=data)

for i, result in enumerate(sentiments):
    print("Sentiment: ", i, result[0])
```

This method will internally split the data into batches of size 16 for increased speed. You can adjust the
batch size using the `batch_size` parameter, e.g., `batch_size=32`.

Alternatively, you can also use the `infer_from_text` method to infer sentiment for a single target:

```python
sentiment = tsc.infer_from_text("I like " ,"Peter", " but I don't like Robert.")
print(sentiment[0])
```

# How to identify a person in a sentence?

In case your data is not separated as shown in the examples above, i.e., in three segments, you will need to identify one (or more) targets first.
How this is done best depends on your project and analysis task but you may, for example, use NER. This [example](https://github.com/fhamborg/NewsMTSC/issues/30#issuecomment-1700645679) shows a simple way of doing so.

# Acknowledgements

Thanks to [Tilman Hornung](https://github.com/t1h0) for adding the batching functionality and various other improvements.

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
