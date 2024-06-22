# Summary

## Outline

### What each file contains
* `mlp.py`
* `e2e.py`
* `asktell.py`
* `seq.py`

### What happens when you import

Right as you `import wazy` the following are imported
* `EnsembleBlockConfig` class in `mlp.py` (which takes a default shape of $(128, 32, 2)$ and no of models = $5$), `AlgConfig` class (contains parameters relevant for training, e.g. batch size, number of epochs, etc.), and certain methods in `mlp.py` e.g. `ensemble_train`, `bayes_opt`, `alg_iter`, `neg_bayesian_ei`, `neg_bayesian_ucb`
* 
* `SeqpropBlock` in `seq.py` which has a method relevant for returning a vector.