from sklearn.model_selection import KFold
from sklearn import  linear_model
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity as cs
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
import csv

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
        allSeed.append(x)
    # for j in range(len(z)):
    #    # allSeed[i] = np.reshape(allSeed[i],(allSeed[0],allSeed[2]))
    #    allSeed[j] = np.reshape(allSeed[j],(734,allSeed[j].shape[2]))
    return allSeed



def splitXVal(samples,labels):
    gkf = GroupKFold(n_splits=5)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    c = 0
    for train_index, test_index in gkf.split(samples,groups=labels):
        a = []
        b = [] 
        c = []
        d = []
        for i in train_index : 
            a.append(samples[i]),b.append(labels[i])
        for i in test_index : 
            c.append(samples[i]),d.append(labels[i])
        x_train.append(a),y_train.append(b),x_test.append(c),y_test.append(d)
    return x_train,x_test,y_train,y_test

def findCoef(x_train,y_train,nN):
    coef = []
    for i in range(len(x_train)):
        n = nN #select top n%
        filtered = [i for i in range (len(x_train[i][0]))] #x_train[0][0] = sample features 
        clf = linear_model.LinearRegression(n_jobs=-1)
        clf.fit(x_train[i],y_train[i])
        co = list(map(lambda x : abs(x),clf.coef_))
        COEF = (sorted(zip(co,[ i for i in range (len(co))]),reverse=True))
        # print(np.max(clf.coef_),np.min(clf.coef_),np.max(co))
        for j in range(len(COEF)):
            if(j < (len(COEF)/100)*n ):
                filtered[COEF[j][1]] = 1 
            else:
                filtered[COEF[j][1]] = 0
        coef.append(filtered)
    return coef

def findDistance(x):
    output = []
    for i in x :
        output.append(cs(i))
    return output

def fixPosition(x):
    output = []
    for i in x :
        for j in range(len(i)):
            for k in range(len(i[j])):
                if(j==k):
                    i[j][k] = 0
        output.append(i)
    return output

def findCloser(x):
    output = []
    for i in x:
        tempSample = []
        for j in range(len(i)):
            sample = i[j]
            close = sorted(zip(sample,[ i for i in range (len(sample))]),reverse=True)[0]
            tempSample.append(close)
        output.append(tempSample)
    return output

def changeIndex2Tag(x,y):
    output = []
    for i in range(len(x)) : #5 group
        temp = []
        for j in range(len(x[i])):#each sample
            a = len(y[i])
            for k in range(a):
                if(x[i][j][1] == k):
                    temp.append(y[i][k])
        output.append(temp)
    return output

def score(rounded,x,y):
    total = []
    for i in range(len(x)) :
        acc = 0
        for j in range(len(x[i])):
            if(x[i][j] == y[i][j]):
                acc = acc+1
        acc = float(acc/len(y[0]))
        acc = round(acc,3)
        total.append(acc)
    #print('{},{},{}'.format(rounded,total , round(np.mean(total),3)))
    return total,round(np.mean(total),3)

def selectFeature(x,coef):
    output = []
    for i in range(len(x)):
        tempSample = []
        for j in range(len(x[i])):
            tempFeature = []
            for k in range(len(coef[i])):
                if(coef[i][k] == 1):
                    tempFeature.append(x[i][j][k])
            tempSample.append(tempFeature)
        output.append(tempSample)
    return output

def xy(x,y,idx):
    outputX = []
    outputY = []
    for index,i in enumerate(y):
        for j in idx:
            if(i==j):
                outputX.append(x[index])
                outputY.append(y[index])
    return outputX,outputY

def forUse(x,y):
    allSeed = []
    for count in range(20):
        a,b = getTrainTestIndex(count)
        allSeed.append([a,b])
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i,j in allSeed:
        a,b = xy(x,y,i)
        c,d = xy(x,y,j)
        x_train.append(a),y_train.append(b),x_test.append(c),y_test.append(d)
    return x_train,y_train,x_test,y_test
    
def extract(x,y):
    X_train = []
    Y_train = []
    X_test  = []
    Y_test = []
    for i,j in zip(x,y) :
        X_train.append(i[0])
        X_test.append(i[1])
        Y_train.append(j[0])
        Y_test.append(j[1])
    return X_train,Y_train,X_test,Y_test
    
features_index_path = 'features_index'
features_path = 'features'

rate = [i for i in range(10,101,10)]
x,y = getFeatureData()
X_train,Y_train,X_test,Y_test = forUse(x,y) #allX[x][y](feature) =  x is seed , y [0,1] 0 is trian feature , 1 is test || ally[x][y](label) x = is seed , y is index
# X_train,Y_train,X_test,Y_test = extract(allX,allY)
for idx,(x,y)in enumerate(zip(X_test[15:],Y_test[15:])):
    a1 = []
    a2 = []
    a3 = [] #cross validation phase
    a4 = []
    a5 = []
    bb = []
    cc = []
    for i in rate :
        #x is samples have n samples for each sample it have features 2048 features 
        #y is label have n number
        x_train = x
        y_train = y 
        # x_test = z
        # Y_test = t
        x_train,x_test,y_train,y_test = splitXVal(x,y)
        #x_train,x_test,y_train,y_test foreach object have 5 group inside.

        coef = findCoef(x_train,y_train,i)
        #z = getFeaturesIndex()
        x_test = selectFeature(x_test,coef)
        #t = mapIndex(x,y,coef)
        distance = findDistance(x_test) #use cosine for every Index
        distance = fixPosition(distance) #fix cosine with itself = 0 
        closer = findCloser(distance) #find the most likely
        closer = changeIndex2Tag(closer,y_test) #change index the closest to label index
        a,b = score(i,closer,y_test) #given score to each sample 
        a1.append(a[0]),a2.append(a[1]),a3.append(a[2]),a4.append(a[3]),a5.append(a[4]),bb.append(b),cc.append(i)
    for i in a1:
        print(i)
    print('------------------------------')
    for i in a2:
        print(i)
    print('------------------------------')
    for i in a3:
        print(i)
    print('------------------------------')
    for i in a4:
        print(i)
    print('------------------------------')
    for i in a5:
        print(i)
    print('------------------------------')
    for i in bb:
        print(i)
    print('------------------------------')
    for i in cc:
        print(i)
    print('------------------------------')

# for i in rate :
#     #x is samples have n samples for each sample it have features 2048 features 
#     #y is label have n number
#     x_train = X_test
#     y_train = Y_test
#     x_test = X_train #test phase
#     y_test = Y_train
#     #x_train,x_test,y_train,y_test = splitXVal(x,y)
#     #x_train,x_test,y_train,y_test foreach object have 5 group inside.

#     coef = findCoef(x_train,y_train,i)
#     #z = getFeaturesIndex()
#     x_test = selectFeature(x_test,coef)
#     #t = mapIndex(x,y,coef)
#     distance = findDistance(x_test) #use cosine for every Index
#     distance = fixPosition(distance) #fix cosine with itself = 0 
#     closer = findCloser(distance) #find the most likely
#     closer = changeIndex2Tag(closer,y_test) #change index the closest to label index
#     a,b = score(i,closer,y_test) #given score to each sample 
#     for i in a : 
#         print(i)
#     print('------------------------------')
