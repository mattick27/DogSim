from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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

def mapIndex(x,y,z) :
    allSeed = []
    for i in range(len(z)):
        #print(x[:,np.where(z[i])])
        allSeed.append(x[:,np.where(z[i])])
    for j in range(len(z)):
       # allSeed[i] = np.reshape(allSeed[i],(allSeed[0],allSeed[2]))
       allSeed[j] = np.reshape(allSeed[j],(734,allSeed[j].shape[2]))
    return allSeed

def forUse(t,y):
    allSeed = []
    for count in range(20):
        a,b = getTrainTestIndex(count)
        allSeed.append([a,b])

    All = []
    for i in range(len(allSeed)):
        Train = []
        Test = []
        for j in range(len(allSeed[0][0])):
            for k in range(len(y)):
                if y[k] == allSeed[i][0][j]:
                    Train.append(k)
        for l in range(len(allSeed[0][1])):
            for m in range(len(y)):
                if y[m] == allSeed[i][1][l]:
                    Test.append(m)
        All.append([Train,Test])
    X_Train = []
    X_Test = []
    aX = []
    for i in range(20):
        for j in (All[i][0]):
            X_Train.append(t[i][j])
        for k in (All[i][1]):
            X_Test.append(t[i][k])
        aX.append([X_Train,X_Test])
        X_Train = []
        X_Test = []
    aY = []
    for i in range(20):
        y_train = []
        y_test = []
        for j in range(len(y[All[i][0]])):
            y_train.append(y[All[i][0][j]])
        for j in range(len(y[All[i][1]])):
            y_test.append(y[All[i][1][j]])

        aY.append([y_train,y_test])
    return aX,aY
        

features_index_path = 'features_index'
features_path = 'features'

x,y = getFeatureData()
z = getFeaturesIndex()
t = mapIndex(x,y,z)
allX,allY = forUse(t,y)
n_cluster = [1,5,15,25]
for seed in range(20):
    for count in (n_cluster):
        number = []
        cosine = []
        numCos=[]
        xtest = []
        acc = 0
        cos = 0
        km = KMeans(n_clusters=count,max_iter=1000,random_state=0)
        KModel = km.fit(allX[seed][0])
        search = KModel.predict(allX[seed][1])
        for i in range(count):
            numGroup = np.where(search == i)
            numGroup = np.reshape(numGroup,(len(numGroup[0])))
            number.append(numGroup)
        for n in range(len(number)):
            for i in (number[n]):
                for j in (number[n]):
                    if i != j :
                        numCos.append(cosine_similarity([allX[seed][1][i]],[allX[seed][1][j]]))
                        #print(numCos)
                    else :
                        numCos.append(-99)
                cosine.append(numCos.index(max(numCos)))
                numCos = []
            for i,c in enumerate(cosine):
                if allY[seed][1][number[n][i]] == allY[seed][1][number[n][c]] :
                    acc += 1
            cosine = []
        print("acc of n = ",count,"seed = ",seed,"= ",(acc/len(allX[seed][1]))*100)


