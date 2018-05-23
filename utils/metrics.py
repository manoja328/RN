import numpy as np

def getaccuracy(true,pred):
    N = len(true)
    return 100*np.sum((true == pred))/N

