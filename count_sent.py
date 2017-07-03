import sys
file = open(sys.argv[1],'r')
file_contents = file.read()
filez = file_contents.split('.')
print(len(filez))
