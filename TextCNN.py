#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: TextCNN.py

# Description:
# Class defining text convolutional neural network using TensorFlow.
#------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf

#------------------------------------------------------------------------------
# A CNN for text classification.
# Uses an embedding layer, followed by a convolutional, max-pooling and softmax
# layer.
#
# Arguments:
# object - class initializer

# Returns:
# None
#------------------------------------------------------------------------------
class TextCNN( object ):
    def __init__(
      self, sequenceLength, numClasses, vocabSize,
      embeddingSize, filtSizes, numFilts, l2RegLambda=0.0 ):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder( tf.int32, [None, sequenceLength], name="input_x" )
        self.input_y = tf.placeholder( tf.float32, [None, numClasses], name="input_y" )
        self.dropoutKeepProb = tf.placeholder( tf.float32, name="dropoutKeepProb" )

        # Keeping track of l2 regularization loss (optional)
        l2Loss = tf.constant( 0.0 )

        # Embedding layer
        with tf.device( '/cpu:0' ), tf.name_scope( "embedding" ):
            self.weight = tf.Variable(
                tf.random_uniform( [vocabSize, embeddingSize], -1.0, 1.0 ),
                name="weight" )
            self.embeddedChars = tf.nn.embedding_lookup( self.weight, self.input_x )
            self.embeddedCharsExpanded = tf.expand_dims( self.embeddedChars, -1 )

        # Create a convolution + maxpool layer for each filter size
        pooledOutputs = []
        for i, filtSize in enumerate( filtSizes ):
            with tf.name_scope( "conv-maxpool-%s" % filtSize ):
                # Convolution Layer
                filtShape = [filtSize, embeddingSize, 1, numFilts]
                weight = tf.Variable( tf.truncated_normal( filtShape, stddev=0.1 ), name="weight" )
                beta = tf.Variable( tf.constant( 0.1, shape=[numFilts]), name="beta" )
                conv = tf.nn.conv2d(
                    self.embeddedCharsExpanded,
                    weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv" )
                # Apply nonlinearity
                reluInst = tf.nn.relu( tf.nn.bias_add( conv, beta ), name="relu" )
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    reluInst,
                    ksize=[1, sequenceLength - filtSize + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool" )
                pooledOutputs.append( pooled )

        # Combine all the pooled features
        numFiltsTotal = numFilts * len( filtSizes )
        self.reluInstPool = tf.concat( pooledOutputs, 3 )
        self.reluInstPoolFlat = tf.reshape( self.reluInstPool, [-1, numFiltsTotal] )

        # Add dropout
        with tf.name_scope( "dropout" ):
            self.reluInstDrop = tf.nn.dropout( self.reluInstPoolFlat, self.dropoutKeepProb )

        # Final (unnormalized) scores and predictions
        with tf.name_scope( "output" ):
            weight = tf.get_variable(
                "weight",
                shape=[numFiltsTotal, numClasses],
                initializer=tf.contrib.layers.xavier_initializer() )
            beta = tf.Variable( tf.constant( 0.1, shape=[numClasses] ), name="beta" )
            l2Loss += tf.nn.l2_loss( weight )
            l2Loss += tf.nn.l2_loss( beta )
            self.scores = tf.nn.xw_plus_b( self.reluInstDrop, weight, beta, name="scores" )
            self.predictions = tf.argmax( self.scores, 1, name="predictions" )

        # CalculateMean cross-entropy loss
        with tf.name_scope( "loss" ):
            losses = tf.nn.softmax_cross_entropy_with_logits( logits=self.scores, labels=self.input_y )
            self.loss = tf.reduce_mean(losses) + l2RegLambda * l2Loss

        # Accuracy
        with tf.name_scope( "accuracy" ):
            correctPredictions = tf.equal( self.predictions, tf.argmax( self.input_y, 1 ) )
            self.accuracy = tf.reduce_mean( tf.cast( correctPredictions, "float" ), name="accuracy" )
