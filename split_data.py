import pdb
import sys
import os
import re

def split_file():
    print("yo")
def clean_folder(files):
    for fp in files:    
        if not (re.search('(bak)\w+',fp)):
            os.remove(fp)
	else:
	    print(fp)
def run(argv):
    split_no = argv[1]
    for root, subdirs, files in os.walk("data/subdata/"):
        clean_folder(files)
run(sys.argv)
