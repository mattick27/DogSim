import numpy as np
import os
from sklearn.utils import shuffle

features_index_path = 'features_index'
features_path = 'features'


def getFeaturesIndex(start=0, stop=20):
    features_index = []
    for x in range(start, stop):
        features_index.append(
            np.load('{}/{}'.format(features_index_path, 'feature_index_Coef_' + str(x) + '.npz'))['arr_0'])
    return np.array(features_index)


def getFeatureData():
    features = []
    ids = []
    for i, className in enumerate(os.listdir(features_path)):
        for file in os.listdir(features_path + '/' + className):
            ids.append(i)
            features.append(
                np.load('{}/{}/{}'.format(features_path, className, file)))
    return np.array(features), np.array(ids)


def getTrainTestIndex(seed):
    a = range(103)
    a = shuffle(a, random_state=seed)
    return a[:50], a[50:]
