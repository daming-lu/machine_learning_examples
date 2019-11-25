import numpy as np
import json
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle
# from word2vec import get_wikipedia_data, find_analogies
from rnn_class.util import get_wikipedia_data
from util import find_analogies
