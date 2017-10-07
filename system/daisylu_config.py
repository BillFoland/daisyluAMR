import platform
import os


def getSystemPath(tag):
    paths = {}
    paths['NNModels']       = '../networks'
    paths['daisylu']        = '../'
    paths['daisyluPython']  = os.getcwd()
 
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
    return paths[tag]

        
