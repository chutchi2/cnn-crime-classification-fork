#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: nlpExp.py

# Description:
# [Description]

# Usage:
# python nlpExp.py
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
twenty_train = fetch_20newsgroups( subset='train', categories=categories, shuffle=True, random_state=42 )
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform( twenty_train.data )

#nlp basic
text_clf = Pipeline( [( 'vect', CountVectorizer() ), ( 'tfidf', TfidfTransformer() ), ( 'clf', MultinomialNB() ),] )
text_clf = text_clf.fit( twenty_train.data, twenty_train.target )
twenty_test = fetch_20newsgroups( subset='test', categories=categories, shuffle=True, random_state=42 )
docs_test = twenty_test.data
predicted2 = text_clf.predict( docs_test )
print( np.mean( predicted2 == twenty_test.target ) )

#svm basic
text_clf = Pipeline( [( 'vect', CountVectorizer() ), ( 'tfidf', TfidfTransformer() ), ( 'clf', SGDClassifier( loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42 ) ),] )
_ = text_clf.fit( twenty_train.data, twenty_train.target )
predicted = text_clf.predict( docs_test )
print( np.mean( predicted == twenty_test.target ) )

#confusion matrix
print( metrics.classification_report( twenty_test.target, predicted, target_names=twenty_test.target_names ) )
print( metrics.confusion_matrix( twenty_test.target, predicted ) )