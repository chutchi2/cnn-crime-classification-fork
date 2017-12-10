#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: trainNBSVM.py

# Description:
# Uses glove embeddings to train Naive-Bayes and Support Vector Machines on
# the GloVe embeddings used for the TextCNN experiment

# Usage:
# python trainNBSVM.py
#------------------------------------------------------------------------------
import dataHelpers
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib import learn
from TextCNN import TextCNN
import yaml

from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#------------------------------------------------------------------------------
# Loads text CNN into a cfg object for referencing in evaluations.
#
# Arguments:
# None

# Returns:
# Configuration
#------------------------------------------------------------------------------
def loadConfig():
    with open( "config.yml", 'r' ) as ymlfile:
        cfg = yaml.load( ymlfile )
    return cfg

#------------------------------------------------------------------------------
class TrainTextCNN():
    def __init__(self, init_value):
        self.cfg = loadConfig()

#------------------------------------------------------------------------------
# Parameters
#
# Arguments:
# cfg - object for referencing in evaluations

# Returns:
# embeddingName, embeddingDim, FLAGS objects
#------------------------------------------------------------------------------
def loadTFParameters( cfg ):
    # Data loading params
    tf.flags.DEFINE_float( "devSamplePercentage", .1, "Percentage of the training data to use for validation" )

    # Model Hyperparameters
    tf.flags.DEFINE_boolean( "enWordEmbed", True, "Enable/disable the word embedding (default: True)" )
    tf.flags.DEFINE_integer( "embedding_dim", 128, "Dimensionality of character embedding (default: 128)" )
    tf.flags.DEFINE_string( "filtSizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')" )
    tf.flags.DEFINE_integer( "numFilts", 128, "Number of filters per filter size (default: 128)" )
    tf.flags.DEFINE_float( "dropoutKeepProb", 0.5, "Dropout keep probability (default: 0.5)" )
    tf.flags.DEFINE_float( "l2RegLambda", 0.0, "L2 regularization lambda (default: 0.0)" )

    # Training parameters
    tf.flags.DEFINE_integer( "batchSize", 64, "Batch Size (default: 64)" )
    tf.flags.DEFINE_integer( "numEpochs", 200, "Number of training epochs (default: 200)" )
    tf.flags.DEFINE_integer( "evaluateEvery", 100, "Evaluate model on dev set after this many steps (default: 100)" )
    tf.flags.DEFINE_integer( "checkpointEvery", 100, "Save model after this many steps (default: 100)" )
    tf.flags.DEFINE_integer( "numCheckpoints", 5, "Number of checkpoints to store (default: 5)" )
    # Misc Parameters
    tf.flags.DEFINE_boolean( "allow_soft_placement", True, "Allow device soft device placement" )
    tf.flags.DEFINE_boolean( "log_device_placement", False, "Log placement of ops on devices" )

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print( "\nParameters:" )
    for attr, value in sorted( FLAGS.__flags.items() ):
        print( "{}={}".format( attr.upper(), value ) )
    print( "" )

    cfg = loadConfig()

    if FLAGS.enWordEmbed and cfg['word_embeddings']['default'] is not None:
        embeddingName = cfg['word_embeddings']['default']
        embeddingDim = cfg['word_embeddings'][embeddingName]['dimension']
    else:
        embeddingDim = FLAGS.embedding_dim

    return embeddingName, embeddingDim, FLAGS

#------------------------------------------------------------------------------
# Prepare datasets for training.
#
# Arguments:
# cfg - object for referencing in evaluations
# FLAGS - TensorFlow flags for referencing model

# Returns:
# x_train, x_dev, y_train, y_dev, vocabProc objects
#------------------------------------------------------------------------------
def prepData( cfg, FLAGS ):
    # Load data
    print( "Loading data..." )
    datasetName = cfg["datasets"]["default"]
    datasets = None
    if datasetName == "mrpolarity":
        datasets = dataHelpers.getMrPolarityDataset( cfg["datasets"][datasetName]["positive_data_file"]["path"],
                                                        cfg["datasets"][datasetName]["negative_data_file"]["path"] )
    elif datasetName == "codydata":
        datasets = dataHelpers.getQuadPolarityDataSet( cfg["datasets"][datasetName]["one_data_file"]["path"],
                                                        cfg["datasets"][datasetName]["two_data_file"]["path"],
                                                        cfg["datasets"][datasetName]["three_data_file"]["path"],
                                                        cfg["datasets"][datasetName]["four_data_file"]["path"] )
    elif datasetName == "20newsgroup":
        datasets = dataHelpers.get20NewsGroupDataset( subset="train",
                                                         categories=cfg["datasets"][datasetName]["categories"],
                                                         shuffle=cfg["datasets"][datasetName]["shuffle"],
                                                         random_state=cfg["datasets"][datasetName]["random_state"] )
    elif datasetName == "localdata":
        datasets = dataHelpers.getLocalDataset( container_path=cfg["datasets"][datasetName]["container_path"],
                                                         categories=cfg["datasets"][datasetName]["categories"],
                                                         shuffle=cfg["datasets"][datasetName]["shuffle"],
                                                         random_state=cfg["datasets"][datasetName]["random_state"] )
    x_text, y = dataHelpers.loadDataLabels( datasets )

    # Build vocabulary
    # Specify cpu for these operations
    #remove the following line to allow for gpu usage
    with tf.device( '/cpu:0' ), tf.name_scope( "embedding" ):
        maxDocLen = max( [len( sentence.split( " " ) ) for sentence in x_text] )
        vocabProc = learn.preprocessing.VocabularyProcessor( maxDocLen )
        x_inter = np.array( list( vocabProc.fit_transform( x_text ) ) )

    # Randomly shuffle data
    np.random.seed( 10 )
    shuffleIndices = np.random.permutation( np.arange( len( y ) ) )
    x_shuffled = x_inter[shuffleIndices]
    y_shuffled = y[shuffleIndices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    devSampleIndex = -1 * int( FLAGS.devSamplePercentage * float( len( y ) ) )
    x_train, x_dev = x_shuffled[:devSampleIndex], x_shuffled[devSampleIndex:]
    y_train, y_dev = y_shuffled[:devSampleIndex], y_shuffled[devSampleIndex:]
    print( "Vocabulary Size: {:d}".format( len( vocabProc.vocabulary_ ) ) )
    print( "Train/Dev split: {:d}/{:d}".format( len( y_train ), len( y_dev ) ) )

    return x_train, y_train

#------------------------------------------------------------------------------
# Opens glove vectors in pipeline parse-able format.
#
# Arguments:
# None

# Returns:
# w2v vectors
#------------------------------------------------------------------------------
def openGlove():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier
    with open( "GloVe/vectors.txt", "rb" ) as lines:
        w2v = { line.split()[0]: np.array( map( float, line.split()[1:] ) )
               for line in lines }
    return w2v

#------------------------------------------------------------------------------
# Defines the interface for the Mean embedding vectorizer.
#
# Arguments:
# object - Class self reference
#------------------------------------------------------------------------------
class MeanEmbeddingVectorizer( object ):

    #------------------------------------------------------------------------------
    # If a word was never seen - it must be at least as infrequent as any of the
    # known words - so the default idf is the max of known idf's.
    #
    # Arguments:
    # self - Class self reference
    # X - Training data
    # y - Training labels

    # Returns:
    # None
    #------------------------------------------------------------------------------
    def __init__( self, word2vec ):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len( word2vec.itervalues().next() )

    #------------------------------------------------------------------------------
    # If a word was never seen - it must be at least as infrequent as any of the
    # known words - so the default idf is the max of known idf's.
    #
    # Arguments:
    # self - Class self reference
    # X - Training data
    # y - Training labels

    # Returns:
    # None
    #------------------------------------------------------------------------------
    def fit( self, X, y ):
        return self

    #------------------------------------------------------------------------------
    # Transforms output matrix into a padded output that can be parsed.
    #
    # Arguments:
    # self - Class self reference
    # X - Training data

    # Returns:
    # None
    #------------------------------------------------------------------------------
    def transform( self, X ):
        return np.array( [
            np.mean( [self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros( self.dim )], axis=0 )
            for words in X
        ] )

#------------------------------------------------------------------------------
# Defines the interface for the TFIDF embedding vectorizer.
#
# Arguments:
# object - Class self reference
#------------------------------------------------------------------------------
class TfidfEmbeddingVectorizer( object ):
    def __init__( self, word2vec ):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len( word2vec.itervalues().next() )

    #------------------------------------------------------------------------------
    # If a word was never seen - it must be at least as infrequent as any of the
    # known words - so the default idf is the max of known idf's.
    #
    # Arguments:
    # self - Class self reference
    # X - Training data
    # y - Training labels

    # Returns:
    # None
    #------------------------------------------------------------------------------
    def fit( self, X, y ):
        tfidf = TfidfVectorizer( analyzer=lambda x: x )
        tfidf.fit( X )

        max_idf = max( tfidf.idf_ )
        self.word2weight = defaultdict(
            lambda: max_idf,
            [( w, tfidf.idf_[i] ) for w, i in tfidf.vocabulary_.items()] )

        return self

    #------------------------------------------------------------------------------
    # Transforms output matrix into a padded output that can be parsed.
    #
    # Arguments:
    # self - Class self reference
    # X - Training data

    # Returns:
    # None
    #------------------------------------------------------------------------------
    def transform( self, X ):
        return np.array( [
                np.mean( [self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros( self.dim )], axis=0 )
                for words in X
            ] )

#------------------------------------------------------------------------------
# Runs training on both the Naive-Bayes and Support Vector Machine models.
#
# Arguments:
# w2v - GloVe input vectors
# x - Training data
# y - Training labels

# Returns:
# None
#------------------------------------------------------------------------------
def run( w2v, x, y ):
    #NB
    etree_w2v_tfidf = Pipeline( [( "word2vec vectorizer", TfidfEmbeddingVectorizer( w2v ) ),( "extra trees", OneVsRestClassifier( MultinomialNB() ) ) ] )
    etree_w2v_tfidf = etree_w2v_tfidf.fit( x, y )
    predicted2 = etree_w2v_tfidf.predict( x )
    print( np.mean( predicted2 == y ) )

    #SVM
    text_clf = Pipeline( [( "word2vec vectorizer", MeanEmbeddingVectorizer( w2v ) ),( 'clf', OneVsRestClassifier( SGDClassifier( loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42 ) ) )] )
    text_clf = text_clf.fit( x, y )
    predicted = text_clf.predict( x )
    print( np.mean( predicted == y ) )

#------------------------------------------------------------------------------
def main( argv ):
    cfg = loadConfig()
    embeddingName, embeddingDim, FLAGS = loadTFParameters( cfg )
    x_train, y_train = prepData( cfg, FLAGS )
    w2v = openGlove()
    run( w2v, x_train, y_train )

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main( sys.argv )