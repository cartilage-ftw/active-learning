# Summary

## Outline

### What happens when you import

Right as you `import wazy` the following are imported
* `EnsembleBlockConfig` class in `mlp.py` (which takes a default shape of $(128, 32, 2)$ and no of models = $5$), `AlgConfig` class (contains parameters relevant for training, e.g. batch size, number of epochs, etc.), and certain methods in `mlp.py` e.g. `ensemble_train`, `bayes_opt`, `alg_iter`, `neg_bayesian_ei`, `neg_bayesian_ucb`
* `EnsembleModel` and `build_naive_e2e` from `e2e.py`
* `utils.py` methods such as `encode_seq`, `differentiable_jax_unirep`, `resample`, `neg_relu`, etc. (see [original file](https://github.com/cartilage-ftw/wazy/blob/master/wazy/__init__.py) for a complete list)
* `SeqpropBlock` in `seq.py` which has a method relevant for returning a vector.
* The `BOAlgorithm` class from `asktell.py` which we instantiated and used most of the time in our code (note: another class called `MCMCAlgorithm` is also imported)