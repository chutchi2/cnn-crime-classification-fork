#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: splitData.py

# Description:
# [Description]

# Usage:
# python splitData.py fracNum dataDir
#------------------------------------------------------------------------------
import pdb
import sys
import os
import re
import subprocess

#------------------------------------------------------------------------------
# [Description]
#
# Arguments:
# [argument] - argument description

# Returns:
# [Description of return]
#------------------------------------------------------------------------------
def splitFile( frac_num, root, fp ):
    bashCommand = "split -n " + str( frac_num ) + " "  + root + "/" + fp
    process = subprocess.Popen( bashCommand.split(), stdout=subprocess.PIPE )
    output, error = process.communicate()
    m = re.search( '(?<=bak)\w+', os.path.join( root, fp ) )

    bashCommand2 = "mv xaa " + root + "/"  + m.group( 0 ) + ".txt"
    process = subprocess.Popen( bashCommand2.split(), stdout = subprocess.PIPE )
    output, error = process.communicate()

    bashCommand3 = "iconv -f us-ascii -t UTF-8"

#------------------------------------------------------------------------------
# [Description]
#
# Arguments:
# [argument] - argument description

# Returns:
# [Description of return]
#------------------------------------------------------------------------------
def cleanFolder( root, files, fracNum ):
    for fp in files:
        if not ( re.search( '(bak)\w+', fp ) ):
            os.remove( os.path.join( root, fp ) )
        else:
            splitFile( fracNum, root, fp )

#------------------------------------------------------------------------------
# [Description]
#
# Arguments:
# [argument] - argument description

# Returns:
# [Description of return]
#------------------------------------------------------------------------------
def run( fracNum, dataDir ):
    for root, subdir, files in os.walk( dataDir = "data/subdata/" ):
        for fp in files:
            if not ( re.search( '(bak)\w+', fp ) ):
                os.remove( os.path.join( root, fp ) )
            else:
                splitFile( fracNum, root, fp )

    bashCommand = "rm x*"
    process = subprocess.Popen( bashCommand.split(), stdout = subprocess.PIPE )
    output, error = process.communicate()

#------------------------------------------------------------------------------
def main( argv ):
   run( argv[1], argv[2] )

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main( sys.argv )
