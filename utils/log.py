import logging

def log(appname='app',filename='temp.log'):
    logger = logging.getLogger(appname)
    logger.setLevel(logging.INFO)
    
    # create the logging file handler
    fh = logging.FileHandler(filename)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # add handler to logger object
    logger.addHandler(fh)
    return logger


def write6metrics(filename,message,acc,peracc,RMSE,RMSEm,MAE,MAEm):
    #accuracy, mean-per class accuracy, RMSE,RMSE^M, MAE,MAE^M
    #both converted to integer
    with open(filename,"a") as resultfile:
        resultfile.write("{}\n".format(message))
        resultfile.write("accuracy: {}\n".format(acc))
        resultfile.write(" per class accuracy: {}\n".format(peracc))
        resultfile.write(" RMSE: {}\n".format(RMSE))
        resultfile.write(" RMSE_m: {}\n".format(RMSEm))
        resultfile.write(" MAE: {}\n".format(MAE))
        resultfile.write(" MAE_m: {}\n".format(MAEm))