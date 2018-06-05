import numpy as np

def getaccuracy(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    N = len(true)
    return 100*np.sum((true == pred))/N

