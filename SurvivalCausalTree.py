import numpy as np
from itertools import islice
import re
import csv
import random
import pandas as pd
# about visualization
'''
import pydotplus
import sys
import os
os.environ["PATH"] += os.pathsep + 'D:/graphviz/bin' # add the path of graphviz
'''


'''1、判断测试集是否为空，是：对树进行塌陷处理 
2、判断树的左右分支是否为树结构，是：根据树当前的特征值、划分值将测试集分为Lset、Rset两个集合； 
3、判断树的左分支是否是树结构：是：在该子集递归调用剪枝过程； 
4、判断树的右分支是否是树结构：是：在该子集递归调用剪枝过程； 
5、判断当前树结构的两个节点是否为叶子节点： 
是: 
a、根据当前树结构，测试集划分为Lset,Rset两部分； 
b、即：测试集在Lset 和 Rset 的正则化后的tao的平方和； 
c、合并后，取叶子节点值为原左右叶子结点的均值。求取测试集在该节点处的tao+1； 
d、比较合并前后tao的大小；若NoMergeError > MergeError,返回合并后的节点；否则，返回原来的树结构； 
否：返回树结构
'''


def loaddata(film): #  load:return 3 values:the feature of patients,gene,and gene's number
    genefeature = []
    alldata = []
    patfeature = []
    line_num = -1
    fr = open(film)
    for line in islice(fr, 0, None):  # skip up 1. about islice https://blog.csdn.net/larykaiy/article/details/82934527 start from the second line
        cueLine = re.split(',|\n ', line.strip())
        #print(cueLine)
        line_num += 1
        if line_num > 0:
            meanLine = list(map(float, cueLine))
            #patLine = list(map(float, cueLine[:3]))
            alldata.append(meanLine)
            #patfeature.append(patLine)
        #(cueLine)
            #fltLine = list(map(float, cueLine[4:]))
            #genefeature.append(fltLine)
        # print(dataMat)
        else:
            genename = cueLine[4:]
    Ndata = np.mat(alldata)
    m, n = np.shape(Ndata)
    b = list(range(0, m))
    slice = RandomSampling(b, 100)
    selecteddata = Ndata[slice, :]
    print('selected data shows here', selecteddata)
    m, n = np.shape(selecteddata)
    print("m={}, n={}". format(m, n))
    patfeature = selecteddata[:, :3]
    genefeature = selecteddata[:, 4:]
    loc_mapping = {value: idx for value, idx in enumerate(genename)}  # dict found by https://blog.csdn.net/shaxiaozilove/article/details/79686816
    #print(loc_mapping)
    return patfeature, genefeature, loc_mapping


def RandomSampling(dataMat, number):  # simple random sample
    try:
        slice = random.sample(dataMat, number)
        print("chose: ",slice)
        return slice
    except:
        print('sample larger than population')


# feature is  the feature we want to separate,value belong to this feature
def binSplitDataSet(patientSet, feature, geneSet, value):
    gsp0 = geneSet[np.nonzero(geneSet[:, feature] > value)[0], :]  # function nonzero: find the location of nozero
    gsp1 = geneSet[np.nonzero(geneSet[:, feature] <= value)[0], :]  # https://blog.csdn.net/lilong117194/article/details/78283358
    psp0 = patientSet[np.nonzero(geneSet[:, feature] > value)[0], :]
    psp1 = patientSet[np.nonzero(geneSet[:, feature] <= value)[0], :]
    return gsp0, gsp1, psp0, psp1


def taoob (patientSet):
    # add a function tao !!
    Weight = patientSet[:, 1]
    Y = patientSet[:, 2]
    pi = np.shape(np.nonzero(Weight == 1)[0])[0]/np.shape(Weight)[0]
    '''
    print("sum(Weight/pi = ", sum(Weight/pi))
    t1 = Weight.T * Y / pi
    print('Weight.T * Y / pi= ', t1)
    '''
    tao = Weight.T * Y / pi / sum(Weight/pi) - (1-Weight).T * Y / (1-pi) / sum((1-Weight) / (1-pi))
    #print("W=1>{} W=0>{} ". format(Weight.T * Y / pi / sum(Weight/pi), (1-Weight).T * Y / (1-pi) / sum((1-Weight) / (1-pi))))
    #print("tao : ",tao)
    return tao


def chooseBestSplit(geneSet, patientSet, TaoValue, errType=taoob, ops=(30, 10)): # value is tao obvious
    tolT = ops[0]
    tolN = ops[1]  # tolS is err we can tolerate  tolN is the at least number of note simple after splitting
    # np.tolist将数组或者矩阵转换成列表     tolist()[0] 列表第一项 对于一维矩阵就是第一个列表  set 集合
    m, n = np.shape(geneSet)
    print("m={},n={}". format(m, n))
    limitT = TaoValue ** 2
    bestT = -np.inf
    bestlT = 0
    bestrT = 0
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n):
        print('featIndex: ', featIndex)
        for splitVal in set(geneSet[:, featIndex].T.A.tolist()[0]):  # must be transformed into vector or it is unhashable
            gsp0, gsp1, psp0, psp1 = binSplitDataSet(patientSet, featIndex, geneSet, splitVal)
            if (np.shape(gsp0)[0] < tolN) or (np.shape(gsp1)[0] < tolN): continue
            newlT = errType(psp0)
            newrT = errType(psp1)
            newT = pow(newlT, 2) + pow(newrT, 2)
            if newT >= bestT:
                bestIndex = featIndex
                bestValue = splitVal
                bestT = newT
                bestlT = newlT
                bestrT = newrT
    if (bestT - limitT) < tolT:
        return None, TaoValue, None, None
    return bestIndex, bestValue, bestlT, bestrT

def prune(tree, testData):  #  the tree would be pruned by a fumula using k and alpha
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



def createTree(geneSet, patientSet, TaoValue, errType=taoob, ops=(30, 10)):
    feat, val, Taol, Taor = chooseBestSplit(geneSet, patientSet, TaoValue)
    if feat == None: return val  # children notes
    retTree = {}
    retTree['spInd'] = feat  # child note
    retTree['spVal'] = val   # feature value
    retTree['TaoVal'] = TaoValue
    glSet, grSet, plSet, prSet = binSplitDataSet(patientSet, feat, geneSet, val, )
    retTree['left'] = createTree(glSet, plSet, Taol, errType, ops)  # recursion
    retTree['right'] = createTree(grSet, prSet, Taor, errType, ops)
    return retTree


if __name__ == '__main__':
    patfeature, genefeature, loc_mapping = loaddata('litdata.csv')
    patMat = np.mat(patfeature)
    genMat = np.mat(genefeature)
    # sample genes   to decrease the required time for running
    print("tao: ", taoob(patMat))
    SurvivalCausalTree = createTree(genMat, patMat, taoob(patMat))
    print(SurvivalCausalTree['right'])







