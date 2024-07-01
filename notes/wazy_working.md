# Summary

## Outline

### What happens when you import

Right as you `import wazy` the following are imported
* `EnsembleBlockConfig` class in `mlp.py` (which takes a default shape of $(128, 32, 2)$ and no of models = $5$), `AlgConfig` class (contains parameters relevant for training, e.g. batch size, number of epochs, etc.), and certain methods in `mlp.py` e.g. `ensemble_train`, `bayes_opt`, `alg_iter`, `neg_bayesian_ei`, `neg_bayesian_ucb`
* `EnsembleModel` and `build_naive_e2e` from `e2e.py`
* `utils.py` methods such as `encode_seq`, `differentiable_jax_unirep`, `resample`, `neg_relu`, etc. (see [original file](https://github.com/cartilage-ftw/wazy/blob/master/wazy/__init__.py) for a complete list)
* `SeqpropBlock` in `seq.py` which has a method relevant for returning a vector.
* The `BOAlgorithm` class from `asktell.py` which we instantiated and used most of the time in our code (note: another class called `MCMCAlgorithm` is also imported)

## BOAlgorithm

* When you instantiate `boa = BOAlgorithm()` the default `__init__()` method in the class accepts two arguments `model_config` and `alg_config`. This `model_config` is exactly the `EnsembleBlockConfig` (containing the number of layers) while `alg_config` contains info such as batch_size, number of training epochs, etc.

In principle, we can define a `model_config` ourselves separately and pass that as an arguemnt to `boa = BOAlgorithm(model_config=config)`, and adjust the network parameters there as well.

* When you call `tell()`, it stores the seq and label, and at the same time calls the `_get_reps()` method (in the same class) to get a "representation" of the sequence. This method doesn't 
    * if your model is pre-trained, it calls `jax_unirep.get_reps()` which returns a multi-dimensional array, which includes the feature vector (from UniRep).
    * Note that the `encode_seq()` method in `utils.py` (which simply returns a naive, one-hot encoded representation) is not called unless you say you don't want to use your pre-trained model.
    * => We can tune this to our choice, create our own method for encoding, and therefore invoke our own "featurization" (if we want to study the effect of that).
* Training is only performed when you call `BOAlgorithm.predict()`, if there are sequences it knows it hasn't trained on.
    * In that case, it either sets up the ensemble (if training has never been called before) by calling `setup_ensemble_train()` from `mlp.py`
        * in `setup_ensemble_train()` there's a parameter `dual=True`. This is to decide the form of the loss function (if true, use adversarial training, otherwise 'naive')

    * The actual training is done by `exec_ensemble_train()` also in `mlp.py`.