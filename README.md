# An active learning scheme for optimizing protein sequences
=====

We demonstrated how the [wazy](https://github.com/ur-whitelab/wazy) package ([Yang _et al_. 2022](https://www.biorxiv.org/content/10.1101/2022.08.05.502972v1.abstract) developed by the White lab) can be trained on protein sequence-property prediction tasks. For training we ran coarse-grained simulations using [HOOMD-blue 2.9.7](https://hoomd-blue.readthedocs.io/en/v2.9.7/) extended with [azplugins](https://github.com/mphowardlab/azplugins). The simulations were run on the [MOGON II](https://mogonwiki.zdv.uni-mainz.de/docs/introduction/what_is_mogon/) computing cluster of JGU Mainz.

Here, we provide the code used for training and the results of the simulations (extracted quantities, e.g. $B_{22}$ or $\Delta G$ for protein sequences), along with the scripts used for generation the simulations and computing aforementioned quantities.

The code presented here was used in the study by Changiarath _et al_. 2024, under review for _Faraday Discussions_.
### Dependencies

Our code builds on [wazy](https://github.com/ur-whitelab/wazy), which performs the featurization using UniRep, and has its own construction for doing bayesian optimization using [MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron) as a surrogate model. We use [localCIDER](https://pappulab.github.io/localCIDER/) for computing descriptors.

Apparently `metapredict` has a lot of dependencies and it'll download large libraries while installing.

```
pip install cython metapredict wazy localcider
```
