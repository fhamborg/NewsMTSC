# Welcome

The files contained in this archive are part of the dataset "NewsMTSC" described in our paper "NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles" published at the EACL 2021.

## Dataset

### Files
The dataset consists of three splits. In practical terms, we suggest to use the files as follows (more detailed information can be found in the paper):

* `train.jsonl` - For **training**.
* `devtest_mt.jsonl` - To evaluate a model's classification performance only on sentences that contain **at least two target mentions**. Note that the mentions were extracted to refer to different persons but in a few cases might indeed refer to the same person since we extracted them automatically.
* `devtest_mt.jsonl` - To evaluate a model's classification performance on a "**real-world**" set of sentences, i.e., the set was created with the objective to resemble real-world distribution as to sentiment and other factors mentioned in the paper.


### Format
Each split is stored in a JSONL file. In JSONL, each line represents one JSON object. In our dataset, each JSON object consists of:

1. `sentence_normalized`: a single sentence
2. `primary_gid`: an identifier that is unique within NewsMTSC
3. `targets`: one or more targets

Each target in `targets` consists of:

1. `Input.gid`: an identifier that is unique within NewsMTSC
2. `from`: the character-based, 0-indexed position of the first character of the target's mention within `sentence_normalized`
3. `to`: the last character of the target's mention
4. `mention`: the text of the mention
5. `polarity`: the sentiment of the sentence concerning the target's mention (2.0 = negative, 4.0 = neutral, 6.0 = positive)
6. `further_mentions` (optional): one or more coreferential mentions of the target within the sentence. Note that these were extracted automatically and thus might be incorrecet or not be complete. Further, our annotators labeled the sentiment concerning the main mention, which - depending on the sentence - might not be identical to the sentiment of the coreferences.

```
{
   "primary_gid":"allsides_1192_476_17_— Judge Neil M. Gorsuch_126_139",
   "sentence_normalized":"But neither side harbored any doubts, based on the judge’s opinions, other writings and the president who nominated him, that Judge Gorsuch would be a reliable conservative committed to following the original understanding of those who drafted and ratified the Constitution.",
   "targets":[
      {
         "Input.gid":"allsides_1192_476_17_— Judge Neil M. Gorsuch_126_139",
         "from":126,
         "to":139,
         "mention":"Judge Gorsuch",
         "polarity":6.0,
         "further_mentions":[
            {
               "from":116,
               "to":119,
               "mention":"him"
            }
         ]
      }
   ]
}
```

## Contact

If you want to get in touch, feel free to contact Felix Hamborg. If you find an issue with the dataset or model or have a question concerning either, please open an issue in the repository.

* Web: [https://felix.hamborg.eu/](https://felix.hamborg.eu/)
* Mail: [felix.hamborg@uni-konstanz.de](mailto:felix.hamborg@uni-konstanz.de)
* Repository: [https://github.com/fhamborg/NewsMTSC](https://github.com/fhamborg/NewsMTSC)


## How to cite

If you use the dataset or parts of it, please cite our paper:

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
