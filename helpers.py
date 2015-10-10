import numpy as np

""" Helpers to calculate error. Currently implemented mean error and rmspe """
def ToZero(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y > 0
    w[ind] = y[ind]
    return w

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(y, yhat):
    y = y
    w = ToWeight(y)
    yhat = ToZero(yhat)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def mean_error(y, yhat):
    return (abs(y - yhat)).mean()


""" Split training dataset into test and validation """
def split_data(data, ratio):
    import pdb; pdb.set_trace()
    return training_data, test_data