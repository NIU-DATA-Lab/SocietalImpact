import os
import pickle
import json
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import feature_extraction as fe
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize
from nltk import PorterStemmer
from nltk import word_tokenize
from statistics import mean, stdev

def has_abstract(j):
    res = j["citation"].get("abstract", False)
    return bool(res)


def random_sample(collection, n, pred=lambda x: True):
    """Get a random sample without replacement from collection, using an optional predicate.
    Args:
        collection (set): collection should be a set of filenames
        n (int): size of sample
        pred (function): predicate f(j) -> Bool where j is dict of json of a datapoint.
    """
    k = 0
    while k < n:
        file_path = random.sample(collection, 1)[0]
        collection.remove(file_path)
        with open(file_path) as f:
            j = json.load(f)
            if pred(j):
                k += 1
                yield j

data = "/home/christian/data/data"

with open(os.path.join(data, "policy.pickle"), "rb") as f:
    policy_collection = pickle.load(f)
with open(os.path.join(data, "non_policy.pickle"), "rb") as f:
    non_policy_collection = pickle.load(f)

len_pol, len_non_pol = len(policy_collection), len(non_policy_collection)

non_abs = []
for d in random_sample(non_policy_collection, 100, pred=has_abstract):
    non_abs.append(d["citation"].get("abstract", ""))

pol_abs = []
for d in random_sample(policy_collection, 100, pred=has_abstract):
    pol_abs.append(d["citation"].get("abstract", ""))

labels = [0] * len(non_abs) + [1] * len(pol_abs)


def evalulate_classifier(docs, labels, clf):
    scores = cross_validate(clf, docs, labels, cv=10, scoring=['precision', 'recall', 'accuracy'])
    print("CROSS VAL SCORES via cross_validate")
    prec_mean = mean(scores['test_precision'])
    prec_stdev = stdev(scores['test_precision'])
    rec_mean = mean(scores['test_recall'])
    rec_stdev = stdev(scores['test_recall'])
    acc_mean = mean(scores['test_accuracy'])
    acc_stdev = stdev(scores['test_accuracy'])
    print("precision (90%% confidence interval): %0.2f (+/- %0.2f)" % (prec_mean, prec_stdev))
    print("recall (90%% confidence interval): %0.2f (+/- %0.2f)" % (rec_mean, rec_stdev))
    print("acccuracy (90%% confidence interval): %0.2f (+/- %0.2f)" % (acc_mean, acc_stdev))


def experiment1(k=500):
    stemmer = PorterStemmer()
    cv = CountVectorizer(stop_words="english")
    docs = non_abs + pol_abs
    ndocs = []

    for doc in docs:
        ndoc = []
        for word in word_tokenize(doc):
            ndoc.append(stemmer.stem(word))
        ndocs.append(" ".join(ndoc))
    docs = ndocs

    cv.fit(docs)
    X_train_counts = cv.transform(docs)
    tf = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf.transform(X_train_counts)

    ch2 = SelectKBest(chi2, k=k)
    ch2.fit(X_train_tf, labels)
    X_train_tf = ch2.fit_transform(X_train_tf, labels)
    return X_train_tf


def experiment2():
    stemmer = PorterStemmer()
    cv = CountVectorizer(stop_words="english")
    docs = non_abs + pol_abs
    ndocs = []

    for doc in docs:
        ndoc = []
        for word in word_tokenize(doc):
            ndoc.append(stemmer.stem(word))
        ndocs.append(" ".join(ndoc))
    docs = ndocs

    cv.fit(docs)
    X_train_counts = cv.transform(docs)
    tf = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf.transform(X_train_counts)
    return X_train_tf


print("Experiment1 with bag of words @ 100")
evalulate_classifier(experiment1(k=100), labels, RandomForestClassifier(n_jobs=-1))

print("Experiment1 with bag of words @ 500")
evalulate_classifier(experiment1(k=500), labels, RandomForestClassifier(n_jobs=-1))

print("Experiment2")
evalulate_classifier(experiment2(), labels, RandomForestClassifier(n_jobs=-1))

