# NERDA [**WIP**] <img src="https://raw.githubusercontent.com/ebanalyse/NERDA/main/logo.png" align="right" height=250/>

![Build status](https://github.com/ebanalyse/NERDA/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/ebanalyse/NERDA/branch/main/graph/badge.svg?token=OB6LGFQZYX)](https://codecov.io/gh/ebanalyse/NERDA)
![PyPI](https://img.shields.io/pypi/v/NERDA.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/NERDA?color=green)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Not only is `NERDA` a mesmerizing muppet-like character. `NERDA` is also
a python package, that offers a slick easy-to-use interface for fine-tuning 
pretrained transformers for Named Entity Recognition
 (=NER) tasks. 

`NERDA`is built on `huggingface` `transformers` and the popular `pytorch`
 framework.

## Installation guide
`NERDA` can be installed from [PyPI](https://pypi.org/project/NERDA/) with 

```
pip install NERDA
```

If you want the development version then install directly from [GitHub](https://github.com/ebanalyse/NERDA).

## Named-Entity Recogntion tasks
Named-entity recognition (NER) (also known as (named) entity identification, 
entity chunking, and entity extraction) is a subtask of information extraction
that seeks to locate and classify named entities mentioned in unstructured 
text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.<sup>[1]</sup>

[1]: https://en.wikipedia.org/wiki/Named-entity_recognition

### Example Task:

**Task** 

Identify person names and organizations in text:

*Jim bought 300 shares of Acme Corp.*

**Solution**

| **Named Entity**   | **Type**              | 
|--------------------|-----------------------|
| 'Jim'              | Person                |
| 'Acme Corp.'       | Organization          |

Read more about NER on [Wikipedia](https://en.wikipedia.org/wiki/Named-entity_recognition).

## Train Your Own `NERDA` Model

*GOAL:* We want to fine-tune an [English ELECTRA](https://huggingface.co/google/electra-small-discriminator) model for NER in English.

Load package.

```python
from NERDA.models import NERDA
```

Instantiate a `NERDA` model (*with default settings*) for the 
[`CoNLL-2003`](https://www.clips.uantwerpen.be/conll2003/ner/) 
English NER data set.

```python
from NERDA.datasets import get_conll_data
model = NERDA(dataset_training = get_conll_data('train'),
              dataset_validation = get_conll_data('valid'),
              transformer = 'google/electra-small-discriminator')
```

The model can then be trained/fine-tuned by invoking the `train` method, e.g.

```python
model.train()
````

**Note**: this will take some time depending on the dimensions of your machine
(if you want to skip training, you can go ahead and use one of the models, 
that we have already precooked for you in stead!).

After the model has been trained, the model can be used for predicting 
named entities in new texts.

```python
# text to identify named entities in.
text = 'Old MacDonald had a farm'
model.predict_text(text)
```
.. It is as simple as that!

Please note, that the `NERDA` model configuration above was instantiated 
with all default settings. You can however customize your `NERDA` model
in a lot of ways:

- Use your own data set (in whatever language you desire)
- Choose whatever transformer you like
- Set all of the hyperparameters for the model
- You can even apply your own Network Architecture 

Read more about advanced usage of `NERDA` in the [detailed documentation](https://ebanalyse.github.io/NERDA/workflow).

## Use a Precooked `NERDA` model

We have precooked a number of `NERDA` models for Danish and English, that you can download 
and use right off the shelf. 

Here is an example.

Instantiate multingual BERT model, that has been finetuned for NER in Danish,
`DA_BERT_ML`.

```python
from NERDA.precooked import DA_BERT_ML()
model = DA_BERT_ML()
```

Down(load) network from web:

```python
model.download_network()
model.load_network()
```

You can now predict named entities in new (Danish) texts

```python
# (Danish) text to identify named entities in.
# = 'Old MacDonald had a farm'
text = 'Jens Hansen har en bondegård'
model.predict_text(text)
```

### List of Precooked Models

The table below shows the precooked `NERDA` models publicly available for download.

| **Model**       | **Language** | **Transformer**   | **Dataset** | **F1-score** |  
|-----------------|--------------|-------------------|---------|-----|
| `DA_BERT_ML`    | Danish       | [Multilingual BERT](https://huggingface.co/bert-base-multilingual-uncased) | [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane) | xx.x  | 
`DA_ELECTRA_DA` | Danish       | [Danish ELECTRA](https://huggingface.co/Maltehb/-l-ctra-danish-electra-small-uncased) | [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane) |yy.y             |
| `EN_BERT_ML`    | English      | [Multilingual BERT](https://huggingface.co/bert-base-multilingual-uncased)| [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | zz.z              |
| `EN_ELECTRA_EN` | Danish       | [English ELECTRA](https://huggingface.co/google/electra-small-discriminator) | [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | pp.p             |

**F1-score** is the micro-averaged F1-score across entity tags and is 
evaluated on the respective tests (that have not been used for training nor
validation of the models).

Note, that we have not spent a lot of time on actually fine-tuning the models,
so there could be room for improvement. If you are able to improve the models,
we will be happy to hear from you and include your `NERDA` model.

## Performance (Obsolete)

The table below summarizes the performance as measured by F1-scores of the model
 configurations, that `NERDA` ships with. 

| **Level**     | **MBERT** | **DABERT** | **ELECTRA** | **XLMROBERTA** | **DISTILMBERT** |
|---------------|-----------|------------|-------------|----------------|-----------------|
| B-PER         | 0.92      | 0.93       | 0.92        | 0.94           | 0.89            |      
| I-PER         | 0.97      | 0.99       | 0.97        | 0.99           | 0.96            |   
| B-ORG         | 0.68      | 0.79       | 0.65        | 0.78           | 0.66            |     
| I-ORG         | 0.67      | 0.79       | 0.72        | 0.77           | 0.61            |   
| B-LOC         | 0.86      | 0.85       | 0.79        | 0.87           | 0.80            |     
| I-LOC         | 0.33      | 0.32       | 0.44        | 0.24           | 0.29            |     
| B-MISC        | 0.73      | 0.74       | 0.61        | 0.77           | 0.70            |     
| I-MISC        | 0.70      | 0.86       | 0.65        | 0.91           | 0.61            |   
| **AVG_MICRO** | 0.81      | 0.85       | 0.79        | 0.86           | 0.78            |      
| **AVG_MACRO** | 0.73      | 0.78       | 0.72        | 0.78           | 0.69            |

## 'NERDA'?
'`NERDA`' originally stands for *'Named Entity Recognition for DAnish'*. However, this
is somewhat misleading, since the functionality is no longer limited to Danish. 
On the contrary it generalizes to all other languages, i.e. `NERDA` supports 
fine-tuning of transformer-based models for NER tasks for any arbitrary 
language.

## Background
`NERDA` is developed as a part of [Ekstra Bladet](https://ekstrabladet.dk/)’s activities on Platform Intelligence in News (PIN). PIN is an industrial research project that is carried out in collaboration between the [Technical University of Denmark](https://www.dtu.dk/), [University of Copenhagen](https://www.ku.dk/) and [Copenhagen Business School](https://www.cbs.dk/) with funding from [Innovation Fund Denmark](https://innovationsfonden.dk/). The project runs from 2020-2023 and develops recommender systems and natural language processing systems geared for news publishing, some of which are open sourced like `NERDA`.

## Read more
The detailed documentation for `NERDA` including code references and
extended workflow examples can be accessed [here](https://ebanalyse.github.io/NERDA/).

## Contact
We hope, that you will find `NERDA` useful.

Please direct any questions and feedbacks to
[us](mailto:lars.kjeldgaard@eb.dk)!

If you want to contribute (which we encourage you to), open a
[PR](https://github.com/ebanalyse/NERDA/pulls).

If you encounter a bug or want to suggest an enhancement, please 
[open an issue](https://github.com/ebanalyse/NERDA/issues).

