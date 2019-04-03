# Implementation of Logistic Regression for pi that means the Tendency to receive treatment
__author__='Ricardo'

from itertools import islice
import re
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics


def SelectFromModel(X, y):
    from sklearn.feature_selection import SelectFromModel
    print(X.shape)
    ListRound = np.logspace(3, -10, 20)
    lsvc = LogisticRegressionCV(multi_class='ovr', cv=3, Cs=ListRound, penalty="l2", solver='lbfgs')
    model = SelectFromModel(lsvc, prefit=False).fit(X, y)
    X_new = model.transform(X)
    FeatureChoose = model.get_support(indices=True)
    print(X_new.shape)
    # print(FeatureChoose)
    return X_new, FeatureChoose


def loadDataSet(FileName, Cv=True, Choose=False):
    data = []
    label = []
    fr = open(FileName)
    for line in islice(fr, 1, None):
        lineArr = re.split(',|\n ', line.strip())
        # print('lineArr: ', lineArr[0])
        datainput = list(map(float, lineArr[:10]))
        labelinput = list(map(int, lineArr[-1]))
        # print('dataInput:', datainput)
        # print('labelInput:', labelinput)
        data.append(datainput)
        label.append(labelinput)
    # dataMat = list(map(float, data))
    # labelMat = list(map(int, label))
    if Choose:
        data_new = SelectFromModel(np.array(data), np.array(label))
    else:
        data_new = np.array(data)
    if Cv:
        X_train, X_test, Y_train, Y_test = train_test_split(data_new, label, test_size=0.3, random_state=0)
        # print('dataMat: ', dataMat)
        # print("X_train: ",X_train)
        print(X_train.shape)
        print(X_test.shape)
        return X_train, X_test, Y_train, Y_test, data_new, np.array(label)
    else:
        return np.array(data), np.array(label)


def FitLinearModel(X, Y, Cv, lowV, highV):
    ListRound = np.logspace(lowV, highV, 20)
    LgCV = LogisticRegressionCV(multi_class='ovr', cv=Cv, Cs=ListRound, penalty="l2", solver='lbfgs')
    # LgCV = LogisticRegressionCV(multi_class='ovr', cv=Cv)
    mode = LgCV.fit(X, Y)
    return mode


def SaveModel(Mod, name):
    from sklearn.externals import joblib
    joblib.dump(Mod, name)  # save
    # joblib.load("logistic_lr.model")


def Evaluate(mode, X, Y, X_test, Y_test):
    from sklearn.metrics import confusion_matrix
    r = mode.score(X, Y)
    print("R值(准确率):", r)
    #print("参数:", mode.coef_)
    #print("截距:", mode.intercept_)
    # print("稀疏化特征比率:%.2f%%" % (np.mean(LgCV.coef_.ravel() == 0) * 100))
    print("=========sigmoid函数转化的值，即：概率p=========")
    # print(mode.predict_proba(X_test))  # sigmoid函数转化的值，即：概率p
    Y_predict = mode.predict(X_test) # predict method could return prob  i*j ith sample equal j label
    labels1 = list(set(Y_predict))
    '''
    print("=============Y_test==============")
    print(list(Y_test))
    print("============Y_predict============")
    print(list(Y_predict))
    '''
    conf_mat1 = confusion_matrix(Y_test, Y_predict, labels=labels1)
    print(conf_mat1)
    # https://blog.csdn.net/u013421629/article/details/78470020
    # 从sklearn.metrics里导入classification_report模块。
    from sklearn.metrics import classification_report
    # 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果。
    print('Accuracy of LR Classifier:', mode.score(X_test, Y_test))
    # 利用classification_report模块获得LogisticRegression其他三个指标的结果。
    print(classification_report(Y_test, Y_predict, target_names=['W=0', 'W=1']))
    print("==================Could you want to save this mode('Yes'or'No')==================")
    Command = input()
    if Command.capitalize() == 'Yes':
        SaveModel(mode, 'LinearLogisticsModel')
    else:
        print('What a shame!')


def CaculatePi(FileName):

    # X_train, X_test, Y_train, Y_test, X_origin, label = loadDataSet(FileName)
    X_origin, Y_train = loadDataSet(FileName, Cv=False)
    X_train_Mat = np.mat(X_origin)
    Y_train_Mat = np.mat(Y_train)
    # X_test_Mat = np.mat(X_test)
    # Y_test_Mat = np.mat(Y_test)
    LogModel = FitLinearModel(X_train_Mat, Y_train_Mat, 3, -10, 5)
    r = LogModel.score(X_train_Mat, Y_train_Mat)
    # print("R值(准确率):", r)
    Prob = LogModel.predict_proba(X_origin)
    # print(Prob)
    # Evaluate(LogModel, X_train_Mat, Y_train_Mat, X_test_Mat, Y_test_Mat)
    Pi = np.array(Prob)
    # print(Pi)
    return Pi


def main():
    newfile = 'simulation1.csv'
    CaculatePi(newfile)


if __name__ == '__main__':
    main()