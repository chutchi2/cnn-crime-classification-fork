#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: countSent.py

# Description:
# Contains the countNumSentences function which will count sentences in text
# files.

# Usage:
# python countSent.py file.txt
#------------------------------------------------------------------------------
import sys

#------------------------------------------------------------------------------
# Takes a txt file as an argument and parses each sentence as a count.
#
# Arguments:
# fileArgument - the name of the file to read on disk

# Returns:
# Number of sentences in input document
#------------------------------------------------------------------------------
def countNumSentences( fileArgument ):
    file = open( fileArgument, 'r' )
    fileContents = file.read()
    pFile = fileContents.split( '.' )
    print( len( pFile ) )

#------------------------------------------------------------------------------
def main( argv ):
    countNumSentences( argv[1] )

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main( sys.argv )
