# NERDA 1.0.0

* NERDA model class is now equipped with functions for saving (loading) weights for a fine-tuned NERDA Network to (from) file. See functions model.save_network() and model.load_network_from_file()

# NERDA 0.9.7

* return confidence scores for predictions of all tokens, e.g. model.predict(x, return_confidence=True).

# NERDA 0.9.6

* compute Precision, Recall and Accuracy (optional) with evaluate_performance().
* improve relative imports inside package.

# NERDA 0.9.5

* ... bugfixes.

# NERDA 0.9.4

* functionality for dynamic quantization, fp32 to fp16, padding parametrized.

# NERDA 0.9.2

* remove precooked DA_BERT_ML_16BIT, include precooked DA_DISTILBERT_ML.

# NERDA 0.9.1

* include 16 bit FP precooked DA_BERT_ML_16BIT.

# NERDA 0.9.0

* Support new versions of `transformers` (4.x) and `torch` 

# NERDA 0.8.7

* BUGFIX: Restrict torch version.
* Do not import datasets as part of Precooked Models.
* Do not load datasets if not provided by user.

# NERDA 0.8.6

* First official release.
