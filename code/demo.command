#!/usr/bin/env python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from model import stopWords, x_train
inp = []
tmp = input("Bitte geben Sie eine Fertigkeit ein / ('beenden' zu beenden): ")
while tmp != 'beenden':
    inp.append(tmp)
    with open('MNB_classifier', 'rb') as f:
        clf = pickle.load(f)

    #feature extraction via vectorization(Vectorizer takes care of cleaning text)
    vectorizer = TfidfVectorizer(stop_words=stopWords, analyzer='word')
    train_vectors = vectorizer.fit_transform(x_train)
    vector = vectorizer.transform(inp)
    prediction = clf.predict(vector)
    for el in prediction:
        print(el)
    tmp = input("Bitte geben Sie eine Fertigkeit ein / ('beenden' zu beenden): ")