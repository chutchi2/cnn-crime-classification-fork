#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: eval.py

# Description:
# Evaluates a trained text CNN model.

# Usage:
# python eval.py
#------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import os
import dataHelpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import sys

#------------------------------------------------------------------------------
# Compute softmax values for each sets of scores in x.
#
# Arguments:
# x - matrix to apply softmax function to for evaluation

# Returns:
# Softmax evaluation of each value in matrix
#------------------------------------------------------------------------------
def softmax( x ):
    if x.ndim == 1:
        x = x.reshape( ( 1, -1 ) )
    maxX = np.max(x, axis=1).reshape( ( -1, 1 ) )
    expX = np.exp( x - maxX )
    return expX / np.sum( expX, axis=1 ).reshape( ( -1, 1 ) )

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
# Loads the TensorFlow parameters into the flags object for referencing in
# evaluation.
#
# Arguments:
# cfg - object for referencing in evaluations

# Returns:
# FLAGS, x_test, datasets, x_raw, y_test objects
#------------------------------------------------------------------------------
def loadTFParameters(cfg):
    # Data Parameters

    # Eval Parameters
    tf.flags.DEFINE_integer( "batchSize", 64, "Batch Size (default: 64)" )
    tf.flags.DEFINE_string( "checkpointDir", "", "Checkpoint directory from training run" )
    tf.flags.DEFINE_boolean( "evalTrain", True, "Evaluate on all training data" )

    # Misc Parameters
    tf.flags.DEFINE_boolean( "allow_soft_placement", True, "Allow device soft device placement" )
    tf.flags.DEFINE_boolean( "log_device_placement", False, "Log placement of ops on devices" )


    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    print( "\nParameters:" )
    for attr, value in sorted( FLAGS.__flags.items() ):
        print( "{}={}".format( attr.upper(), value ) )
    print( "" )

    datasets = None

    # CHANGE THIS: Load data. Load your own data here
    datasetName = cfg["datasets"]["default"]
    if FLAGS.evalTrain:
        if datasetName == "mrpolarity":
            datasets = dataHelpers.getMrPolarityDataset( cfg["datasets"][datasetName]["positive_data_file"]["path"],
                                                 cfg["datasets"][datasetName]["negative_data_file"]["path"] )
        elif datasetName == "20newsgroup":
            datasets = dataHelpers.get20NewsGroupDataset( subset="test",
                                                  categories=cfg["datasets"][datasetName]["categories"],
                                                  shuffle=cfg["datasets"][datasetName]["shuffle"],
                                                  random_state=cfg["datasets"][datasetName]["random_state"] )
        elif datasetName == "codydata":
            datasets = dataHelpers.getQuadPolarityDataSet( cfg["datasets"][datasetName]["one_data_file"]["path"],
                                                            cfg["datasets"][datasetName]["two_data_file"]["path"],
                                                            cfg["datasets"][datasetName]["three_data_file"]["path"],
                                                            cfg["datasets"][datasetName]["four_data_file"]["path"] )
        x_raw, y_test = dataHelpers.loadDataLabels( datasets )
        y_test = np.argmax( y_test, axis=1 )
        print( "Total number of test examples: {}".format( len( y_test ) ) )
    else:
        if datasetName == "mrpolarity":
            datasets = { "target_names": ['positive_examples', 'negative_examples'] }
            x_raw = [ "a masterpiece four years in the making", "everything is off." ]
            y_test = [ 1, 0 ]
        else:
            datasets = { "target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'] }
            x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                     "I am in the market for a 24-bit graphics card for a PC"]
            y_test = [2, 1]

    # Map data into vocabulary
    #vocabPath = os.path.join( FLAGS.checkpointDir,"..", "vocab" )
    vocabPath = "/home/cody/cnn-environment/cnn-crime-classification-fork/runs/1512931063/vocab"
    vocabProc = learn.preprocessing.VocabularyProcessor.restore( vocabPath )
    x_test = np.array( list( vocabProc.transform( x_raw ) ) )

    return FLAGS, x_test, datasets, x_raw, y_test

#------------------------------------------------------------------------------
# Evaluation
#
# Arguments:
# FLAGS - TensorFlow flags for referencing model
# x_test - input values to test against model

# Returns:
# allPredicitions, allProbabilities objects
#------------------------------------------------------------------------------
def evaluate( FLAGS, x_test ):
    print( "\nEvaluating...\n" )

    #checkpointFile = tf.train.latest_checkpoint( FLAGS.checkpointDir )
    checkpointFile = "/home/cody/cnn-environment/cnn-crime-classification-fork/runs/1512931063/checkpoints/model-103200"
    graph = tf.Graph()
    with graph.as_default():
        sessionConf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement )
        sess = tf.Session( config=sessionConf )
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph( "{}.meta".format( checkpointFile ) )
            saver.restore( sess, checkpointFile )

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name( "input_x" ).outputs[0]
            # input_y = graph.get_operation_by_name( "input_y" ).outputs[0]
            dropoutKeepProb = graph.get_operation_by_name( "dropoutKeepProb" ).outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name( "output/scores" ).outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name( "output/predictions" ).outputs[0]

            # Generate batches for one epoch
            batches = dataHelpers.batchIter( list( x_test ), FLAGS.batchSize, 1, shuffle=False )

            # Collect the predictions here
            allPredicitions = []
            allProbabilities = None

            for x_testBatch in batches:
                batchPredictionScores = sess.run( [predictions, scores], { input_x: x_testBatch, dropoutKeepProb: 1.0 } )
                allPredicitions = np.concatenate( [allPredicitions, batchPredictionScores[0]] )
                probabilities = softmax( batchPredictionScores[1] )
                if allProbabilities is not None:
                    allProbabilities = np.concatenate( [allProbabilities, probabilities] )
                else:
                    allProbabilities = probabilities
    return allPredicitions, allProbabilities
#------------------------------------------------------------------------------
# Print accuracy if y_test is defined
#
# Arguments:
# y_test - all labels for data in test data
# allPredicitions - all predicted labels for the test data based off the model
# datasets - dataset to use in classification report

# Returns:
# None
#------------------------------------------------------------------------------
def showYTest( y_test, allPredicitions, datasets ):
    if y_test is not None:
        correctPredicitons = float( sum( allPredicitions == y_test ) )
        print( "Total number of test examples: {}".format( len( y_test ) ) )
        print( "Accuracy: {:g}".format( correctPredicitons / float( len( y_test ) ) ) )
        print( metrics.classification_report( y_test, allPredicitions, target_names=datasets['target_names'] ) )
        print( metrics.confusion_matrix( y_test, allPredicitions ) )

#------------------------------------------------------------------------------
# Saves evaluations to a csv file based off model used
#
# Arguments:
# x_raw - all training and testing data
# allPredicitions - all predictions from evaluation
# allProbabilities - Each probability for a given prediction based off softmax
# FLAGS -TensorFlow flags for referencing model.

# Returns:
# CSV file of evaluations
#------------------------------------------------------------------------------
def saveEvals( x_raw, allPredicitions, allProbabilities, FLAGS):
    predictionsHumanReadable = np.column_stack( ( np.array( x_raw ),
                                                  [int(prediction) for prediction in allPredicitions],
                                                  [ "{}".format( probability ) for probability in allProbabilities] ) )
    outPath = os.path.join( FLAGS.checkpointDir, "..", "prediction.csv" )
    print( "Saving evaluation to {0}".format( outPath ) )
    with open( outPath, 'w' ) as path:
        csv.writer( path ).writerows( predictionsHumanReadable )

#------------------------------------------------------------------------------
def main( argv ):
    cfg = loadConfig()
    FLAGS, x_test, datasets, x_raw, y_test = loadTFParameters( cfg )
    allPredicitions, allProbabilities = evaluate( FLAGS, x_test )
    showYTest( y_test, allPredicitions, datasets )
    saveEvals( x_raw, allPredicitions, allProbabilities , FLAGS )

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main( sys.argv )