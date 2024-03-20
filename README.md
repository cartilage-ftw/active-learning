# active-learning
=======
# An active learning scheme for optimizing protein sequences

### Dependencies

This version of our code builds on [wazy](https://github.com/ur-whitelab/wazy), which performs the featurization using UniRep, and has its own construction for doing bayesian optimization using [MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron) as a surrogate model. We use [localCIDER](https://pappulab.github.io/localCIDER/) for computing descriptors.

Apparently `metapredict` has a lot of dependencies and it'll download large libraries while installing.

```
pip install cython metapredict wazy localcider
```
