import numpy as np
import matplotlib.pyplot as plt

from future.utils import iteritems
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from sklearn.manifold import TSNE


def dist1(a, b):
    return np.linalg.norm(a - b)
def dist2(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
embedding = []
idx2word = []

word2idx = {}
my_idx2word = {}
# 6 billion corpus, 50 dimension
with open('../large_files/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  row_idx = 0
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
    embedding.append(vec)
    idx2word.append(word)
    word2idx[word] = row_idx
    row_idx += 1
    if row_idx > 2000:
        break


print('Found %s word vectors.' % len(word2vec))
embedding = np.array(embedding)
V, D = embedding.shape

# map back to word in plot
my_idx2word = {v: k for k, v in iteritems(word2idx)}

# transformer = TfidfTransformer()
# A = transformer.fit_transform(A.T).T

v_law = word2vec['law']
v_laws = word2vec['laws']
v_stock = word2vec['stock']
v_stocks = word2vec['stocks']

v_measure = word2vec['measure']
v_measures = word2vec['measures']


law_diff = v_law - v_laws
stock_diff = v_stock - v_stocks
measure_diff = v_measure - v_measures

print(dist2(law_diff, stock_diff))
print(dist2(law_diff, measure_diff))
print(dist2(stock_diff, measure_diff))

chosen = ['japan', 'japanese', 'china', 'chinese', 'law', 'laws', 'stock', 'stocks', 'measure', 'measures',
          'italy', 'italian', 'france', 'french']

import ipdb;ipdb.set_trace()
# plot the data in 2-D
tsne = TSNE()
Z = tsne.fit_transform(embedding)
plt.scatter(Z[:, 0], Z[:, 1])
for i in range(V):
    try:
        word = my_idx2word[i].encode("utf8").decode("utf8")
        if word not in chosen:
            continue
        plt.annotate(s=word, xy=(Z[i, 0], Z[i, 1]))
    except:
        print("bad string:", idx2word[i])
plt.draw()

plt.show() # pause script until plot is closed
