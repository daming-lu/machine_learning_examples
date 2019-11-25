import numpy as np
import json
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# (nlp_udemy) DamingLu:nlp_class2 ludaming$ [master]$ declare -x PYTHONPATH="/Users/ludaming/Baidu_USA_DamingLu/udemy/nlp/machine_learning_examples/"

from datetime import datetime
from sklearn.utils import shuffle
from word2vec import get_wikipedia_data, find_analogies
