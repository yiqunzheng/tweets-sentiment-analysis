import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import cPickle as pickle

def get_tweets(filename):
    with open(filename) as f:
        return [line[1:] for line in f]
    
def get_labels(filename):
    with open(filename) as f:
        return np.array([int(line[0]) for line in f])

class Model(object):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          max_features=5000)
        self.model = LogisticRegression()

    def fit(self, tweets, labels):
        X = self.vectorizer.fit_transform(tweets).toarray()
        self.model.fit(X, labels)
        return self

    def predict(self, tweets):
        X = self.vectorizer.transform(tweets).toarray()
        return self.model.predict(X)

def build_model(data_filename, model_filename):
    tweets = get_tweets(data_filename)
    labels = get_labels(data_filename)
    model = Model().fit(tweets,labels)
    if model_filename:
        with open(model_filename, 'w') as f:
            pickle.dump(model, f)
    return model

if __name__ == '__main__':
    build_model('training.txt', 'model.pkl')
    