#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: splitData.py

# Description:
# Contains the necessary functions to split all files in a certain directory.

# Usage:
# python splitData.py fracNum dataDir
#------------------------------------------------------------------------------
import os
import re
import subprocess
import sys

#------------------------------------------------------------------------------
# Splits a text file using unix bash commands into a fraction of the original
# size. Leaves original file intact.
#
# Arguments:
# divideByNum - the fraction of the size to divide the file by
# root - the root directory string of the execution environment
# fp - the file pointer to the file which will be split

# Returns:
# None
#------------------------------------------------------------------------------
def splitFile( divideByNum, root, fp ):
    bashCommand = "split -n " + str( divideByNum ) + " "  + root + "/" + fp
    process = subprocess.Popen( bashCommand.split(), stdout=subprocess.PIPE )
    output, error = process.communicate()

    m = re.search( '(?<=bak)\w+', os.path.join( root, fp ) )
    bashCommand2 = "mv xaa " + root + "/"  + m.group( 0 ) + ".txt"
    process = subprocess.Popen( bashCommand2.split(), stdout = subprocess.PIPE )
    output, error = process.communicate()

    #bashCommand3 = "iconv -f us-ascii -t UTF-8"

#------------------------------------------------------------------------------
# Takes a file list and decides to remove the file or to split the file.
#
# Arguments:
# root - argument description
# files - a list of all files to search over
# fracNum - the fraction of the size to divide the file by

# Returns:
# None
#------------------------------------------------------------------------------
def cleanFolder( root, files, fracNum ):
    for fp in files:
        if not ( re.search( '(bak)\w+', fp ) ):
            os.remove( os.path.join( root, fp ) )
        else:
            splitFile( fracNum, root, fp )

#------------------------------------------------------------------------------
# Takes a file as an argument and parses each sentence as a count.
#
# Arguments:
# files - a list of all files to search over

# Returns:
# Number of sentences in input document
#------------------------------------------------------------------------------
def countNumSentences( root, files ):
    for fp in files:
        if not ( re.search( '(bak)\w+', fp ) ):
            file = open( os.path.join( root, fp ), 'r' )
            fileContents = file.read()
            pFile = fileContents.split( '.' )
            print ( len( pFile ) )
            return len( pFile )

#------------------------------------------------------------------------------
# Runs the necessary building and cleaning to split files.
#
# Arguments:
# fracNum - the fraction of the size to divide the file by
# dataDir - location of files to split

# Returns:
# None
#------------------------------------------------------------------------------
def run( fracNum, dataDir="data/subdata/" ):
    for root, subdir, files in os.walk( dataDir ):
        cleanFolder( root, files, fracNum )
        countNumSentences( root, files )

    bashCommand = "rm xaa xab xac xad xae xaf xag xah xai xaj"
    process = subprocess.Popen( bashCommand.split(), stdout = subprocess.PIPE )
    output, error = process.communicate()

#------------------------------------------------------------------------------
def main( argv ):
    run( argv[1], argv[2] )

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main( sys.argv )
