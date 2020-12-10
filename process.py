import pickle as pk
from ToolScripts.TimeLogger import log
import os
# import scipy.sparse as sp

def loadData2(datasetStr, cv):
    assert datasetStr == "Tianchi_time"
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    with open(DIR + '/pvTime.csv'.format(cv), 'rb') as fs:
        pvTimeMat = pk.load(fs)
    with open(DIR + '/cartTime.csv'.format(cv), 'rb') as fs:
        cartTimeMat = pk.load(fs)
    with open(DIR + '/favTime.csv'.format(cv), 'rb') as fs:
        favTimeMat = pk.load(fs)
    with open(DIR + '/buyTime.csv'.format(cv), 'rb') as fs:
        buyTimeMat = pk.load(fs)
    with open(DIR + "/test_data.csv".format(cv), 'rb') as fs:
        test_data = pk.load(fs)
    interatctMat = ((pvTimeMat + cartTimeMat + favTimeMat + buyTimeMat) != 0) * 1
    with open(DIR + "/trust.csv".format(cv), 'rb') as fs:
        trust = pk.load(fs)
    return interatctMat, trust, test_data
    


def loadData(datasetStr, cv):
    if datasetStr == "Tianchi_time":
        return loadData2(datasetStr, cv)
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    log(DIR)
    with open(DIR + '/train.csv', 'rb') as fs:
        trainMat = pk.load(fs)
    with open(DIR + '/test_data.csv', 'rb') as fs:
        testData = pk.load(fs)
    with open(DIR + '/valid_data.csv', 'rb') as fs:
        validData = pk.load(fs)
    with open(DIR + '/train_time.csv', 'rb') as fs:
        trainTimeMat = pk.load(fs)
    with open(DIR + '/trust.csv', 'rb') as fs:
        trustMat = pk.load(fs)
    return trainMat, trustMat, testData
    
    