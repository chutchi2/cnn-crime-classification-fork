#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: [File Name]

# Description:
# [Description]

# Usage:
# python [filename].py [arguments]
#------------------------------------------------------------------------------
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
#X_train_counts.shape()
#count_vect.vocabulary_.get(u'algorithm')

# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
# docs_new = ['Hindu is life', 'OpenGL on the GPU is fast']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = clf.predict(X_new_tfidf)
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))

#nlp basic
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted2 = text_clf.predict(docs_test)
print(np.mean(predicted2 == twenty_test.target))

#svm basic
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

#confusion matrix
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))