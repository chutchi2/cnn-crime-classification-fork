#! /usr/bin/env python
#------------------------------------------------------------------------------
# Filename: [File Name]

# Description:
# [Description]

# Usage:
# python [filename].py [arguments]
#------------------------------------------------------------------------------
import sys

#------------------------------------------------------------------------------
# [Description]
#
# Arguments:
# [argument] - argument description

# Returns:
# [Description of return]
#------------------------------------------------------------------------------
def countNumSentences(argv):
    file = open(argv[1],'r')
    file_contents = file.read()
    filez = file_contents.split('.')
    print(len(filez))

def main(argv):
    countNumSentences(argv)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
