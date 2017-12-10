#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: train.py

# Description:
# Trains a text CNN model

# Usage:
# python train.py
#------------------------------------------------------------------------------
import dataHelpers
import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
from TextCNN import TextCNN
import time
import sys
import yaml

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
# Loads the TensorFlow parameters into the flags object for referencing in
# evaluation.
#
# Arguments:
# cfg - object for referencing in evaluations

# Returns:
# embeddingName, embeddingDim, FLAGS objects
#------------------------------------------------------------------------------
def loadTFParameters( cfg ):
    # Parameters
    # ==================================================
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

    return x_train, x_dev, y_train, y_dev, vocabProc
#------------------------------------------------------------------------------
# Train one batch of data.
#
# Arguments:
# cnn - TextCNN class object
# FLAGS - TensorFlow flags for referencing model
# x_batch - a segment of the x_text data to train on
# y_batch - a segment of the y_text labels to train on
# sess - the TensorFlow session object
# trainOp - a TensorFlow defined variable
# globalStep - a TensorFlow defined variable
# trainSummaryOp - a TensorFlow defined variable
# writer - Default:None

# Returns:
# None
#------------------------------------------------------------------------------
def trainStep( cnn, FLAGS, x_batch, y_batch, sess, trainOp, globalStep, trainSummaryOp, writer=None ):
    feedDict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropoutKeepProb: FLAGS.dropoutKeepProb
    }
    _, step, summaries, loss, accuracy = sess.run(
        [trainOp, globalStep, trainSummaryOp, cnn.loss, cnn.accuracy],
        feedDict )
    timeStr = datetime.datetime.now().isoformat()
    print( "{}: step {}, loss {:g}, acc {:g}".format( timeStr, step, loss, accuracy ) )
    writer.add_summary( summaries, step )

#------------------------------------------------------------------------------
# Evaluates model on a dev set.
#
# Arguments:
# cnn - TextCNN class object
# x_batch - a segment of the x_text data to train on
# y_batch - a segment of the y_text labels to train on
# sess - the TensorFlow session object
# globalStep - a TensorFlow defined variable
# devSummaryOp - a TensorFlow defined variable
# writer - Default:None

# Returns:
# None
#------------------------------------------------------------------------------
def devStep( cnn, x_batch, y_batch, sess, globalStep, devSummaryOp, writer=None ):
    feedDict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropoutKeepProb: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [globalStep, devSummaryOp, cnn.loss, cnn.accuracy],
        feedDict )
    timeStr = datetime.datetime.now().isoformat()
    print( "{}: step {}, loss {:g}, acc {:g}".format( timeStr, step, loss, accuracy ) )
    if writer:
        writer.add_summary( summaries, step)

#------------------------------------------------------------------------------
# Train a text CNN on a datset.
#
# Arguments:
# embeddingName - Either GloVe or word2vec
# FLAGS - TensorFlow flags for referencing model
# x_train - data for training
# x_dev - data for validation
# y_train - labels for training
# y_dev - labels for validation
# vocabProc - Based on the maximum document length a processor is provided
# embeddingDim - Embedding dimension size defined in config.yml
# cfg - object for referencing in evaluations

# Returns:
# None
#------------------------------------------------------------------------------
def train( embeddingName, FLAGS, x_train, x_dev, y_train, y_dev, vocabProc , embeddingDim, cfg ):
    with tf.Graph().as_default():
        sessionConf = tf.ConfigProto(
          allow_soft_placement = FLAGS.allow_soft_placement,
          log_device_placement = FLAGS.log_device_placement )
        sess = tf.Session( config = sessionConf )

        with sess.as_default():
            cnn = TextCNN(
                sequenceLength=x_train.shape[1],
                numClasses=y_train.shape[1],
                vocabSize=len( vocabProc.vocabulary_ ),
                embeddingSize=embeddingDim,
                filtSizes=list( map( int, FLAGS.filtSizes.split( "," ) ) ),
                numFilts=FLAGS.numFilts,
                l2RegLambda=FLAGS.l2RegLambda )

            # Define Training procedure
            globalStep = tf.Variable( 0, name="globalStep", trainable=False )
            optimizer = tf.train.AdamOptimizer( 1e-3 )
            gradsAndVars = optimizer.compute_gradients( cnn.loss )
            trainOp = optimizer.apply_gradients( gradsAndVars, global_step=globalStep )

            # Keep track of gradient values and sparsity (optional)
            gradSummaries = []
            for grad, var in gradsAndVars:
                if grad is not None:
                    gradHistSummary = tf.summary.histogram( "{}/grad/hist".format( var.name ), grad )
                    sparsitySummary = tf.summary.scalar( "{}/grad/sparsity".format( var.name ), tf.nn.zero_fraction( grad ) )
                    gradSummaries.append( gradHistSummary )
                    gradSummaries.append( sparsitySummary )
            gradSummariesMerged = tf.summary.merge( gradSummaries )

            # Output directory for models and summaries
            timestamp = str( int( time.time() ) )
            outDir = os.path.abspath( os.path.join( os.path.curdir, "runs", timestamp ) )
            print( "Writing to {}\n".format( outDir ) )

            # Summaries for loss and accuracy
            lossSummary = tf.summary.scalar( "loss", cnn.loss )
            accSummary = tf.summary.scalar( "accuracy", cnn.accuracy )

            # Train Summaries
            trainSummaryOp = tf.summary.merge( [lossSummary, accSummary, gradSummariesMerged] )
            trainSummaryDir = os.path.join( outDir, "summaries", "train" )
            trainSummaryWriter = tf.summary.FileWriter( trainSummaryDir, sess.graph )

            # Dev summaries
            devSummaryOp = tf.summary.merge( [lossSummary, accSummary] )
            devSummaryDir = os.path.join( outDir, "summaries", "dev" )
            devSummaryWriter = tf.summary.FileWriter( devSummaryDir, sess.graph )

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpointDir = os.path.abspath( os.path.join( outDir, "checkpoints" ) )
            checkpointPrefix = os.path.join( checkpointDir, "model" )
            if not os.path.exists( checkpointDir ):
                os.makedirs( checkpointDir )
            saver = tf.train.Saver( tf.global_variables(), max_to_keep=FLAGS.numCheckpoints )

            # Write vocabulary
            vocabProc.save( os.path.join( outDir, "vocab" ) )

            # Initialize all variables
            sess.run( tf.global_variables_initializer() )
            if FLAGS.enWordEmbed and cfg['word_embeddings']['default'] is not None:
                vocabulary = vocabProc.vocabulary_
                initW = None
                if embeddingName == 'word2vec':
                    # load embedding vectors from the word2vec
                    print( "Load word2vec file {}".format( cfg['word_embeddings']['word2vec']['path'] ) )
                    initW = dataHelpers.loadWord2VecEmbeddings( vocabulary,
                                                                         cfg['word_embeddings']['word2vec']['path'],
                                                                         cfg['word_embeddings']['word2vec']['binary'] )
                    print( "word2vec file has been loaded" )

                elif embeddingName == 'glove':
                    # load embedding vectors from the glove
                    print( "Load glove file {}".format( cfg['word_embeddings']['glove']['path'] ) )
                    initWeight = dataHelpers.loadGloveEmbeddings( vocabulary,
                                                                      cfg['word_embeddings']['glove']['path'],
                                                                      embeddingDim )
                    print( "glove file has been loaded\n" )
                sess.run( cnn.weight.assign( initWeight ) )

            # Generate batches
            batches = dataHelpers.batchIter(
                list( zip( x_train, y_train ) ), FLAGS.batchSize, FLAGS.numEpochs )

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip( *batch )
                trainStep( cnn, FLAGS, x_batch, y_batch, sess, trainOp, globalStep, trainSummaryOp, writer=trainSummaryWriter )
                currentStep = tf.train.global_step( sess, globalStep )
                if currentStep % FLAGS.evaluateEvery == 0:
                    print( "\nEvaluation:" )
                    devStep( cnn, x_dev, y_dev, sess, globalStep, devSummaryOp, writer=devSummaryWriter )
                    print( "" )
                if currentStep % FLAGS.checkpointEvery == 0:
                    path = saver.save( sess, checkpointPrefix, global_step=currentStep )
                    print( "Saved model checkpoint to {}\n".format( path ) )

#------------------------------------------------------------------------------
def main( argv ):
    cfg = loadConfig()
    embeddingName, embeddingDim, FLAGS = loadTFParameters( cfg )
    x_train, x_dev, y_train, y_dev, vocabProc = prepData( cfg, FLAGS )
    train( embeddingName, FLAGS, x_train, x_dev, y_train, y_dev, vocabProc, embeddingDim, cfg )


#------------------------------------------------------------------------------
if __name__ == "__main__":
    main( sys.argv )