import pickle
import numpy as np

from sklearn.metrics import mean_squared_error


def smse(y_true, y_pred):
    """
    Standerdised Mean Squared Error
    """
    return mean_squared_error(y_true, y_pred) / np.var(y_true)


def pickle_save(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
