# daisyluAMR

## Source code to accompany the paper presented at ACL 2017:

## Abstract Meaning Representation Parsing using LSTM Recurrent Neural Networks, by William Foland and James H. Martin.

http://www.aclweb.org/anthology/P17-1043

Installation
---
Tested on 

macOS 10.12 (Sierra) and Linux 14.04 using 

Python 2.7.13 :: Anaconda 2.3.0.

Tensorflow 1.0.1

Keras 2.0.4

networkx 1.10

## Before cloning this repository, 

install git lfs (large file storage)
https://git-lfs.github.com/

Then clone.

download Illinois wikifier
https://cogcomp.org/page/software_view/Wikifier

install keras with tensorflow
https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/

install smatch 
https://github.com/snowblink14/smatch

modify system/daisylu_config.py to contain paths for your setup

    # Modify these settings for your system (mac or linux or both)
    if (platform.system() == 'Darwin'): #mac settings
        # path to python 2.7 executable
        paths['python']       = '/Users/bill_foland/anaconda/bin/python '  
        # path to Illinois wikifier
        paths['Wikifier2013'] = '/Users/bill_foland/Wikifier2013'
        # command line to run smatch
        paths['smatchCommand'] = 'python ../../smatch_2.0/smatch.py'
    else: # linux settings
        # path to python 2.7 executable
        paths['python']       = 'python '   
        # path to Illinois wikifier
        paths['Wikifier2013'] = '/home/bill/Wikifier2013'
        # command line to run smatch
        paths['smatchCommand'] = 'python /home/bill/smatch_2.0/smatch.py'



Test
----

A set of files is provided to check installation and to demonstrate the format for
sentence files and golden (eg. human generated) AMRs used to check parser output.  These are very short excerpts from the Little Prince corpus from https://amr.isi.edu/download.html.

To run the parse test using defaults:

    cd system
    python daisylu_main.py

install sqlite, pandas, or other missing python packages on your system based on runtime errors

The system should run all five networks and create a file containing the AMR for each sentence, and then use smatch to compare parser output with golden output.  This should result in a smatch score of around 0.6519 

Parse
----

to parse your own sentences, first create a file with a format similar to the enclosed example and specify it on the command line with -i.  The golden file is specified with -g, and the output filename with -o.

    cd system
    python daisylu_main.py -i <sentenceFn> -o <outputFn> -g <goldenFn>



Example
----
DEFT2014T12 Results can be found using the -noWiki option,
first merge all the the test files into "merged.txt" then run

python daisylu_main.py -i merged.txt -o merged.amr -g merged.txt -noWiki
result is  Document F-score: 0.6852




