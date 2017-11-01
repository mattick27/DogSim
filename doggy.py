from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn import tree
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
allX,allY = forUse(t,y) #allX[x][y](feature) =  x is seed , y [0,1] 0 is trian feature , 1 is test || ally[x][y](label) x = is seed , y is index
n_cluster = [1,5,15,25]
data = []
km = KMeans(n_clusters=5,max_iter=1000,random_state=0)
KModel = km.fit(allX[0][0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(allX[0][0],KModel.labels_)
group = clf.predict(allX[0][1]) 

#-----------------------------------------------------------------------
testWithLabel = [[],[],[]] #testWithLabel[x] x = [0,1,2] 0 is groupID , 1 is Label , 2 is Feature
testWithLabel[0].append(group)
testWithLabel[1].append(allY[0][1])
testWithLabel[2].append(allX[0][1])


#-----------------------------------------------------------------------
xG0 = [[],[],[],[],[],[]] #xG0[x] x = [0,1,2,3,4,5] 0 is Label , 1 is Feature , 2 is index , 3 is cosine sim , 4 is the highest cosine sim , 5 is index of the highest value
xG1 = [[],[],[],[],[],[]]
xG2 = [[],[],[],[],[],[]]
xG3 = [[],[],[],[],[],[]]
xG4 = [[],[],[],[],[],[]]

for i in range (len(testWithLabel[0][0])):
    if testWithLabel[0][0][i] == 0 :
        xG0[0].append(testWithLabel[1][0][i])
        xG0[1].append(testWithLabel[2][0][i])
        xG0[2].append(i)
    elif testWithLabel[0][0][i] == 1 :
        xG1[0].append(testWithLabel[1][0][i])
        xG1[1].append(testWithLabel[2][0][i])
        xG1[2].append(i)
    elif testWithLabel[0][0][i] == 2 :
        xG2[0].append(testWithLabel[1][0][i])
        xG2[1].append(testWithLabel[2][0][i])
        xG2[2].append(i)
    elif testWithLabel[0][0][i] == 3 :
        xG3[0].append(testWithLabel[1][0][i])
        xG3[1].append(testWithLabel[2][0][i])
        xG3[2].append(i)
    elif testWithLabel[0][0][i] == 4 :
        xG4[0].append(testWithLabel[1][0][i])
        xG4[1].append(testWithLabel[2][0][i])
        xG4[2].append(i)

for i in range(379): # find cosine in group
    if testWithLabel[0][0][i] == 0 :
        xG0[3].append(cs(xG0[1],testWithLabel[2][0][i]))
    elif testWithLabel[0][0][i] == 1 :
        xG1[3].append(cs(xG1[1],testWithLabel[2][0][i]))        
    elif testWithLabel[0][0][i] == 2 :
        xG2[3].append(cs(xG2[1],testWithLabel[2][0][i]))
    elif testWithLabel[0][0][i] == 3 :
        xG3[3].append(cs(xG3[1],testWithLabel[2][0][i]))
    elif testWithLabel[0][0][i] == 4 :
        xG4[3].append(cs(xG4[1],testWithLabel[2][0][i]))        

for i in range(len(xG0[3])): #Clean cosine with itself
    xG0[3][i][i] = 0
for i in range(len(xG1[3])):
    xG1[3][i][i] = 0
for i in range(len(xG2[3])):
    xG2[3][i][i] = 0
for i in range(len(xG3[3])):
    xG3[3][i][i] = 0
for i in range(len(xG4[3])):
    xG4[3][i][i] = 0
    
for i in range(len(xG0[3])): #Find the highest value
    xG0[4].append(max(xG0[3][i]))
for i in range(len(xG1[3])):
    xG1[4].append(max(xG1[3][i]))
for i in range(len(xG2[3])):
    xG2[4].append(max(xG2[3][i]))
for i in range(len(xG3[3])):
    xG3[4].append(max(xG3[3][i]))
for i in range(len(xG4[3])):
    xG4[4].append(max(xG4[3][i]))

for i in range(len(xG0[3])): #Find ID of the highest value
        count = 0        
        for k in range(len(xG0[3][0])):
            if xG0[3][i][k][0] == xG0[4][i][0]:
                count = count + 1
                if count == 1:
                    xG0[5].append(xG0[2][k])
                
for i in range(len(xG1[3])): #Find ID of the highest value
        count = 0    
        for k in range(len(xG1[3][0])):
            if xG1[3][i][k][0] == xG1[4][i][0]:
                count = count + 1
                if count == 1:
                    xG1[5].append(xG1[2][k])
                
for i in range(len(xG2[3])): #Find ID of the highest value
        count = 0
        for k in range(len(xG2[3][0])):
            if xG2[3][i][k][0] == xG2[4][i][0]:
                count = count + 1
                if count == 1:
                    xG2[5].append(xG2[2][k])
                
for i in range(len(xG3[3])): #Find ID of the highest value
        count = 0
        for k in range(len(xG3[3][0])):
            if xG3[3][i][k][0] == xG3[4][i][0]:
                count = count + 1
                if count == 1:
                    xG3[5].append(xG3[2][k])
                
for i in range(len(xG4[3])): #Find ID of the highest value
        count = 0
        for k in range(len(xG4[3][0])):
            if xG4[3][i][k][0] == xG4[4][i][0]:
                count = count + 1
                if count == 1:
                    xG4[5].append(xG4[2][k])

acc1 = 0
acc2 = 0 
acc3 = 0
acc4 = 0
acc5 = 0
for i in range(len(xG0[5])): #Find acc
    if xG0[0][i] == testWithLabel[1][0][xG0[5][i]]:
        acc1 = acc1 + 1
for i in range(len(xG1[5])): #Find acc
    if xG1[0][i] == testWithLabel[1][0][xG1[5][i]]:
        acc2 = acc2 + 1
for i in range(len(xG2[5])): #Find acc
    if xG2[0][i] == testWithLabel[1][0][xG2[5][i]]:
        acc3 = acc3 + 1
for i in range(len(xG3[5])): #Find acc
    if xG3[0][i] == testWithLabel[1][0][xG3[5][i]]:
        acc4 = acc4 + 1
for i in range(len(xG4[5])): #Find acc
    if xG4[0][i] == testWithLabel[1][0][xG4[5][i]]:
        acc5 = acc5 + 1

acc1 = acc1/len(xG0[5])
acc2 = acc2/len(xG1[5])
acc3 = acc3/len(xG2[5])
acc4 = acc4/len(xG3[5])
acc5 = acc5/len(xG4[5])
print(acc1,acc2,acc3,acc4,acc5)
print((acc1+acc2+acc3+acc4+acc5)/5)

