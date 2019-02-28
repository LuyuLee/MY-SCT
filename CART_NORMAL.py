import numpy as np


def loadDataSet(fileNmae):
    dataMat = []
    fr = open(fileNmae)
    for line in fr.readlines():
        cueLine = line.strip().split('\t')  #  sep = 'tab'
        '''fltLine = []
        for i in cueLine:
            fltLine.append(float(i))'''
        fltLine = list(map(float, cueLine)) #  map all element to float and map is not compatible for int
        # map(function,var)  all var in function  here change str into float for adding
        dataMat.append(fltLine)
    # print('dataMat: ',dataMat)
    return dataMat


# feature is  the feature we want to separate,value belong to this feature
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :] # function nonzero: find the location of nozero
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :] # https://blog.csdn.net/lilong117194/article/details/78283358
    # print("feature: ", feature)
    return mat0, mat1


# mean
def regLeaf (dataSet):
    return np.mean(dataSet[:, -1])


# SSe
def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]  # tolS is err we can tolerate  tolN is the at least number of note simple after splitting
    # np.tolist将数组或者矩阵转换成列表     tolist()[0] 列表第一项 对于一维矩阵就是第一个列表  set 集合
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)    # The only feature is leaf note
    m, n = np.shape(dataSet)
    print("m={},n={}". format(m,n))
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        print('featIndex: ', featIndex)
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):  # must be transformed into vector or it is unhashable
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    return bestIndex, bestValue



def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val  # children notes
    retTree = {}
    retTree['spInd'] = feat  # child note
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # recursion
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__=='dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNomMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) +\
            sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNomMerge:
            print("Merging")
            return treeMean
        else: return tree
    else: return tree


data = loadDataSet('ex2.txt')
myMat = np.mat(data)
print('myMat= ', myMat)
retTree = createTree(myMat, ops=(0, 1))
print(retTree)
print(retTree['spInd'])
dataTest = loadDataSet('ex2test.txt')
myMatTest = np.mat(dataTest)
Newtree = prune(retTree, myMatTest)
print(Newtree)