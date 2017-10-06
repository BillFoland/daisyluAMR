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
        paths['smatchCommand'] = 'python ../../smatch_2.0/smatch.py'
    else: # linux settings
        paths['python']       = 'python '   
        paths['Wikifier2013'] = '/home/bill/Wikifier2013'
        paths['smatchCommand'] = 'python /home/bill/smatch_2.0/smatch.py'
    return paths[tag]

        
