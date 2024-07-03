# Python program to convert text
# file to JSON
  
  
import json
  
  
# the file to be converted to 
# json format

import inspect
import sys
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

names = ['relation', 'entity'] 

for name in names:
    typename = name + '2id'
    filename = typename +'.txt'
    jsonname = typename + '.json'
    # filename = 'relation2id.txt'
    # filename = 'TLogic/TLogic-main/data/ICEWS18/relation2id.txt'
    filename = currentdir +'/' +filename

    
    # dictionary where the lines from
    # text will be stored
    dict1 = {}
    
    # creating dictionary
    with open(filename) as fh:
    
        for line in fh:
    
            # reads each line and trims of extra the spaces 
            # and gives only the valid words
            command, description, *_ = line.strip().split('\t') #line.strip().split(None, 1)
    
    
            dict1[command] = description #.strip()
    
    # creating json file
    # the JSON file is named as test1
    out_file = open(currentdir+"/" + jsonname, "w")
    json.dump(dict1, out_file, indent = 4, sort_keys = False)
    out_file.close()
