from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score,     recall_score
import pandas as pd
import numpy as np

def get_tweets(filename):
    with open(filename) as f:
        return [line[1:] for line in f]

def get_labels(filename):
    with open(filename) as f:
        return np.array([int(line[0]) for line in f])

def get_vectorizer(tweets, num_features=5000):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
    return vect.fit(tweets)

def run_model(Model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), f1_score(y_test, y_predict), precision_score(y_test, y_predict), recall_score(y_test, y_predict)     

def compare_models(tweets, labels, models):
    tweets_train, tweets_test, y_train, y_test = train_test_split(tweets, labels)

    print "-----------------------------"
    run_test(models, tweets_train, tweets_test, y_train, y_test)
    print "-----------------------------"

def run_test(models, tweets_train, tweets_test, y_train, y_test):
    vect = get_vectorizer(tweets_train)
    X_train = vect.transform(tweets_train).toarray()
    X_test = vect.transform(tweets_test).toarray()

    print "acc\tf1\tprec\trecall"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec = run_model(Model, X_train, X_test, y_train, y_test)
        print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)


tweets = get_tweets('training.txt')
labels = get_labels('training.txt')
print "distribution of labels:"
for i, count in enumerate(np.bincount(labels)):
    print "%d: %d" % (i, count)
models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
          RandomForestClassifier]
compare_models(tweets, labels, models)

