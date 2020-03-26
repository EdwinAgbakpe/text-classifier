import json
from sklearn.model_selection import train_test_split
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics  import classification_report, confusion_matrix, accuracy_score
import pickle

#Load file
json_file = open('tech_soft_none.json')
data = json.load(json_file)
sents = []
labels = []
#split into input and labels
for p in data['data']:
    sents.append(p['text'])
    labels.append(p['label'])


#split into train and test data(3:1)
x_train, x_test, y_train, y_test = train_test_split(sents, labels, test_size = 0.25, random_state = 1000)


stopWords = stopwords.words('german')

#feature extraction via vectorization(Vectorizer takes care of cleaning text)
vectorizer = TfidfVectorizer(stop_words=stopWords, analyzer='word')
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)


#train model
clf = MultinomialNB().fit(train_vectors, y_train)

#save model
with open('MNB_classifier', 'wb') as picklefile:
    pickle.dump(clf,picklefile)

#testing
# test_predicted = clf.predict(test_vectors)
# train_predicted = clf.predict(train_vectors)


# #metric evaluations
# print ("Accuracy Scores:")
# print("Train:")
# print(accuracy_score(y_train, train_predicted))
# print("Test:")
# print(accuracy_score(y_test, test_predicted))

# print("Classification Reports:")
# print("Train:")
# print(classification_report(y_train, train_predicted))
# print("Test:")
# print(classification_report(y_test, test_predicted))

# print("Confusion Matrices:")
# print("Train:")
# print(confusion_matrix(y_train, train_predicted))
# print("Test:")
# print(confusion_matrix(y_test, test_predicted))
