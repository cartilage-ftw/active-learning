import pandas as pd
import numpy as np
import wazy
import jax
import matplotlib.pyplot as plt
import time

import warnings
# I wanted to suppress the stupid warnings in the terminal output
warnings.filterwarnings("default", category=UserWarning)
# but this didn't get rid of them -_-

plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']

key = jax.random.PRNGKey(0)

bo_alg = wazy.BOAlgorithm()
bo_alg.tell(key, 'DSSPEAPAEPPKDVPHDWLYSYVFLTHHPADFLR', 0.1531)
#print(bo_alg.predict(key, 'YSPTSPSYSPTSPSYSPTSPS'))
print(bo_alg.ask(key, "max", length=5))
