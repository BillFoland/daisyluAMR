
import platform
import os


def getSystemPath(tag):
    paths = {}
    paths['NNModels']       = '../networks'
    paths['daisylu']        = '../'
    paths['daisyluPython']  = os.getcwd()
    
    # mac settings
    if (platform.system() == 'Darwin'):
        paths['python']       = '/Users/bill_foland/anaconda/bin/python '  
        paths['Wikifier2013'] = '/Users/bill_foland/Desktop/CU/PhD/Wikifier2013'
    else: # linux settings
        paths['python']       = 'python '   
        paths['Wikifier2013'] = '/home/bill/Wikifier2013'
    return paths[tag]

        
