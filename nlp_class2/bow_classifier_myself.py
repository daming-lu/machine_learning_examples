# redo classifier using GloVe

from __future__ import print_function, division
import pdb;pdb.set_trace()
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


# data from https://www.cs.umb.edu/~smimarog/textmining/datasets/
train = pd.read_csv('../large_files/r8-train-all-terms.txt',
                    header=None, sep='\t')
test = pd.read_csv('../large_files/r8-test-all-terms.txt',
                   header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']

print('here')
import ipdb;ipdb.set_trace()
class GloveVectorizer:
    def __init__(self):
        # load in pre-trained word vectors
        word2vec = {}
        embedding = []
        idx2word = []
        with open('../large_files/glove.6B/glove.6B.50d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)

        print('Found %s word vectors.' % len(word2vec))

        # save for later
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)

            if len(vecs) > 0:
                # this sentence has at least one word that has word2vec
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1

        print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
        # must be empty, otherwise one label will have no corresponding content
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


vectorizer = GloveVectorizer()

Xtrain = vectorizer.fit_transform(train.content)
Ytrain = train.label

Xtest = vectorizer.transform(test.content)
Ytest = test.label

model = ExtraTreesClassifier(n_estimators=200)
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

