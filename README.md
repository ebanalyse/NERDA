# NERDA <img src="https://raw.githubusercontent.com/ebanalyse/NERDA/main/logo.png" align="right" height=250/>

![Build status](https://github.com/ebanalyse/NERDA/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/ebanalyse/NERDA/branch/main/graph/badge.svg?token=OB6LGFQZYX)](https://codecov.io/gh/ebanalyse/NERDA)
![PyPI](https://img.shields.io/pypi/v/NERDA.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/NERDA?color=green)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

'NERDA' is not only a mesmerizing muppet-like character. 'NERDA' is also
a python package, that offers a complete framework for fine-tuning of
pretrained transformer-models for Named Entity Recognition (=NER) tasks.

## Installation guide
```
pip install NERDA
```

## NER tasks
Named Entity Recognition (NER) tasks are all about identifying and 
extracting names of named entitites from natural language texts. 

Read more about NER on [Wikipedia](https://en.wikipedia.org/wiki/Named-entity_recognition).

## Performance

The table below summarizes the performance (=F1-scores) of the model
 configurations, that `NERDA` ships with.

| **Level** | **MBERT** | **DABERT** | **ELECTRA**  |
|---------------------------------------------------|
| B-PER     | 0.92      | 0.93       | 0.92         |
| I-PER     | 0.97      | 0.99       | 0.97         |
| B-ORG     | 0.68      | 0.79       | 0.65         |
| I-ORG     | 0.67      | 0.79       | 0.72         |
| B-LOC     | 0.86      | 0.85       | 0.79         |
| I-LOC     | 0.33      | 0.32       | 0.44         |
| B-MISC    | 0.73      | 0.74       | 0.61         |
| I-MISC    | 0.70      | 0.86       | 0.65         |
| AVG_MICRO | 0.81      | 0.85       | 0.79         |
| AVG_MACRO | 0.73      | 0.78       | 0.72         |

## '`NERDA`'?
'`NERDA`' originally stands for *'Named Entity Recognition for DAnish'*. However, this
is somewhat misleading, since the functionality is no longer limited to Danish. 
On the contrary it generalizes to all other languages, i.e. NERDA supports 
fine-tuning of transformer-based models for NER tasks for any arbitrary 
language.

## Read more
The documentation for `NERDA` including code references and
examples can be accessed [here](https://ebanalyse.github.io/NERDA/).

## Contact
We hope, that you will find `NERDA` useful.

Please direct any questions and feedbacks to
[us](mailto:lars.kjeldgaard@eb.dk)!

If you want to contribute (which we encourage you to), open a
[PR](https://github.com/ebanalyse/NERDA/pulls).

If you encounter a bug or want to suggest an enhancement, please 
[open an issue](https://github.com/ebanalyse/NERDA/issues).

