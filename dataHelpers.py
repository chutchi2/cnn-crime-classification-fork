#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: dataHelpers.py

# Description:
# Library of functions for loading machine learning data.

# Usage:
# python dataHelpers.py
#------------------------------------------------------------------------------
import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
import nltk.data
import pdb
import sys

#------------------------------------------------------------------------------
# Tokenization/string cleaning for all datasets except for SST. Original taken
# from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#
# Arguments:
# string - string value to clean of punctuation marks.

# Returns:
# Cleaned string as a lowercase value
#------------------------------------------------------------------------------
def cleanStr( string ):
    string = re.sub( r"[^A-Za-z0-9(),!?\'\`]", " ", string )
    string = re.sub( r"\'s", " \'s", string )
    string = re.sub( r"\'ve", " \'ve", string )
    string = re.sub( r"n\'t", " n\'t", string )
    string = re.sub( r"\'re", " \'re", string )
    string = re.sub( r"\'d", " \'d", string )
    string = re.sub( r"\'ll", " \'ll", string )
    string = re.sub( r",", " , ", string )
    string = re.sub( r"!", " ! ", string )
    string = re.sub( r"\(", " \( ", string )
    string = re.sub( r"\)", " \) ", string )
    string = re.sub( r"\?", " \? ", string )
    string = re.sub( r"\s{2,}", " ", string )

    return string.strip().lower()

#------------------------------------------------------------------------------
# Generates a batch iterator for a dataset.
#
# Arguments:
# data - argument description
# batchSize - the total batch size for each per epoch to train with
# numEpochs - number of epochs to train with
# shuffle - boolean to tell the iterator to shuffle inputs

# Returns:
# A generator that when called will create the training iterator
#------------------------------------------------------------------------------
def batchIter( data, batchSize, numEpochs, shuffle=True ):
    data = np.array( data )
    dataSize = len( data )
    numBatchesPerEpoch = int( ( len( data ) - 1 ) / batchSize ) + 1

    for epoch in range( numEpochs ):
        # Shuffle the data at each epoch
        if shuffle:
            shuffleIndices = np.random.permutation( np.arange( dataSize ) )
            shuffledData = data[shuffleIndices]

        else:
            shuffledData = data

        for batchNum in range( numBatchesPerEpoch ):
            startIndex = batchNum * batchSize
            endIndex = min( ( batchNum + 1 ) * batchSize, dataSize )
            yield shuffledData[startIndex:endIndex]

#------------------------------------------------------------------------------
# Retrieve data from 20 newsgroups
#
# Arguments:
# subset - train, test or all
# categories - List of newsgroup name
# shuffle - shuffle the list or not
# random_state - seed integer to shuffle the dataset

# Returns:
# Data and labels of the newsgroup
#------------------------------------------------------------------------------
def get20NewsGroupDataset( subset='train', categories=None, shuffle=True,
                              random_state=42 ):
    datasets = fetch_20newsgroups(subset=subset, categories=categories,
                                  shuffle=shuffle, random_state=random_state)
    return datasets

#------------------------------------------------------------------------------
# Loads MR polarity data from files, splits the data into words and generates labels.
#
# Arguments:
# positiveDataFile - positive data file for bipolar training
# negativeDataFile - negative data file for bipolar training

# Returns:
# Returns split sentences and labels.
#------------------------------------------------------------------------------
def getMrPolarityDataset( posDataFile, negDataFile ):
    # Load data from files
    posExample = list( open( posDataFile, "r" ).readlines() )
    posExample = [s.strip() for s in posExample]

    negExample = list( open( negDataFile, "r" ).readlines() )
    negExample = [s.strip() for s in negExample]

    datasets = dict()
    datasets['data'] = posExample + negExample

    target = [0 for x in posExample] + [1 for x in negExample]

    datasets['target'] = target
    datasets['target_names'] = ['posExample', 'negExample']

    return datasets

#------------------------------------------------------------------------------
# Loads data from files, splits the data into words and generates labels.
#
# Arguments:
# dataFileOne - one of four files to train one against rest classification
# dataFileTwo - two of four files to train one against rest classification
# dataFileThree - three of four files to train one against rest classification
# dataFileFour - four of four files to train one against rest classification

# Returns:
# Returns split sentences and labels.
#------------------------------------------------------------------------------
def getQuadPolarityDataSet( dataFileOne, dataFileTwo, dataFileThree,
                           dataFileFour ):
    # Load data from files
    oneExamples = list( open( dataFileOne, "r" ).readlines() )
    oneExamples = [s.strip() for s in oneExamples]

    twoExamples = list( open( dataFileTwo, "r" ).readlines() )
    twoExamples = [s.strip() for s in twoExamples]

    threeExamples = list( open( dataFileThree, "r" ).readlines() )
    threeExamples = [s.strip() for s in threeExamples]

    fourExamples = list( open( dataFileFour, "r" ).readlines() )
    fourExamples = [s.strip() for s in fourExamples]

    datasets = dict()
    datasets['data'] = oneExamples + twoExamples + threeExamples + fourExamples

    target = [0 for x in oneExamples] + [1 for x in twoExamples] + [2 for x in threeExamples] + [3 for x in fourExamples]

    datasets['target'] = target
    datasets['target_names'] = ['oneExamples', 'twoExamples',
                                'threeExamples', 'fourExamples']
    return datasets

#------------------------------------------------------------------------------
# Load text files with categories as subfolder names. Individual samples are
# assumed to be files stored a two levels folder structure.
#
# Arguments:
# containerPath - The path of the container
# categories - List of classes to choose, all classes are chosen by default
#              (if empty or omitted)
# shuffle - shuffle the list or not
# randomState - seed integer to shuffle the dataset

# Returns:
# Data and labels of the dataset
#------------------------------------------------------------------------------
def getLocalDataset( containerPath=None, categories=None,
                            load_content=True, encoding='utf-8',
                            shuffle=True, randomState=42 ):
    datasets = load_files( container_path=containerPath, categories=categories,
                           load_content=load_content, shuffle=shuffle,
                           encoding=encoding,random_state=randomState )
    return datasets

#------------------------------------------------------------------------------
# Load data and labels
#
# Arguments:
# datasets - argument description

# Returns:
# Data and labels of the dataset
#------------------------------------------------------------------------------
def loadDataLabels( datasets ):
    # Split by words
    x_init = datasets['data']
    x_text = []

    tokenizer = nltk.data.load( 'tokenizers/punkt/english.pickle' )
    for token in tokenizer.tokenize( str(x_init) ):
        x_text.append( cleanStr( token ) )

    # Generate labels
    labels = []
    for i in range( len( x_text ) ):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append( label )

    y = np.array( labels )

    return [x_text, y]

#------------------------------------------------------------------------------
# Load embeddingVectors from the word2vec
#
# Arguments:
# vocabulary - argument description
# filename - argument description
# binary - argument description

# Returns:
# embeddingVectors
#------------------------------------------------------------------------------
def loadWord2VecEmbeddings( vocabulary, filename, binary ):
    encoding = 'utf-8'
    with open( filename, "rb" ) as pFile:
        header = pFile.readline()
        vocabSize, vecSize = map( int, header.split() )
        # initial matrix with random uniform
        embeddingVectors = np.random.uniform( -0.25, 0.25,
                                             ( len( vocabulary ), vecSize ) )

        if binary:
            binLen = np.dtype( 'float32' ).itemsize * vecSize
            for lineNum in range( vocabSize ):
                word = []
                while True:
                    ch = pFile.read( 1 )
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError( "unexpected end of input; is count incorrect or file otherwise damaged?" )
                    if ch != b'\n':
                        word.append(ch)
                word = str( b''.join( word ), encoding=encoding, errors='strict' )
                idx = vocabulary.get( word )
                if idx != 0:
                    embeddingVectors[idx] = np.fromstring( pFile.read( binLen ), dtype='float32' )
                else:
                    pFile.seek( binLen, 1 )

        else:
            for lineNum in range(vocabSize):
                line = pFile.readline()
                if line == b'':
                    raise EOFError( "unexpected end of input; is count incorrect or file otherwise damaged?" )
                parts = str( line.rstrip(), encoding=encoding, errors='strict' ).split( " " )
                if len( parts ) != vecSize + 1:
                    raise ValueError( "invalid vector on line %s (is this really the text format?)" % ( lineNum ) )
                word, vector = parts[0], list( map( 'float32', parts[1:] ) )
                idx = vocabulary.get( word )
                if idx != 0:
                    embeddingVectors[idx] = vector

        pFile.close()

        return embeddingVectors

#------------------------------------------------------------------------------
# Load embeddingVectors from the glove. Initial matrix with random uniform
#
# Arguments:
# vocabulary - GloVe generated vocabulary file
# filename - the file object to parse as GloVe
# vecSize - size of GloVe vectors to use

# Returns:
# Word embedding vectors for training
#------------------------------------------------------------------------------
def loadGloveEmbeddings( vocabulary, filename, vecSize ):
    embeddingVectors = np.random.uniform( -0.25, 0.25,
                                         ( len( vocabulary ), vecSize ) )
    pFile = open( filename )
    for line in pFile:
        values = line.split()
        word = values[0]
        vector = np.asarray( values[1:], dtype="float32" )
        idx = vocabulary.get( word )
        if idx != 0:
            embeddingVectors[idx] = vector
    pFile.close()

    return embeddingVectors
