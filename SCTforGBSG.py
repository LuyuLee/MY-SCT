# Implementation of Regression Tree using CART algorithm
__author__ = 'Ricardo'

import sys
import csv
import math
import copy
import time
import numpy as np
from collections import Counter
from numpy import *  # genfromtxt
import LogRegresionforGBSG as lr

# reading from the file using numpy genfromtxt
def load_csv(file):
    X = genfromtxt(file, delimiter=",", dtype=str)
    # print(X)
    return (X)


def generate_set(x, Pi, CrossTest = 1):
    import numpy as np
    import random
    # print(Pi.shape)
    Feature_name = x[0, 2:9]
    # print('Feature_name :', Feature_name)
    # print(X[1:, 1:2].shape, X[1:, 9:].shape)
    # print("type: ", x[1])
    X = np.concatenate((x[1:], Pi), 1)
    # print("new X: ", X)
    random.seed(152)
    np.random.shuffle(X)
    # print("type: ", X[1])
    Label = np.concatenate((X[:, 1:2], X[:, 9:]), 1)
    # print(Label.shape, 'Label :', Label)
    Y = Label[:, 2]
    # print(Y.shape, "Y is", Y)
    # print(Y)
    j = Y.reshape(len(Y), 1)
    # print(j.shape, "J is", j)
    new_X = X[:, 2:9]
    # normalizing the data step
    # normalized_X = normalize(new_X)
    # print("Normal X",normalized_X)
    # https://blog.csdn.net/qq_38150441/article/details/80488800
    final_X = np.concatenate((new_X, Label), axis=1)
    # print("np ", final_X)
    X = final_X
    size_of_rows = X.shape[0]
    # test data size is 10%
    num_test = round(0.3 * (X.shape[0]))
    start = 0
    end = num_test
    test_attri_list = []
    test_class_names_list = []
    training_attri_list = []
    training_class_names_list = []
    # ten fold cross-validation
    if CrossTest == 0:
        X_training = X[:, :-5]
        X_training = X_training.astype(np.float)
        # print("X:", X_training)
        y_training = X[:, -5:]
        y_train = y_training.astype(np.float)
        # print("Y:", y_train)
        training_attri_list.append(X_training)
        training_class_names_list.append(y_train)
        return Feature_name, None, None, training_attri_list, training_class_names_list
    for i in range(10):
        X_test = X[start:end, :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        # X_training = X[:start,:]+ X[end: , :]
        y_test = X_test[:, -5:]
        # flatten https://blog.csdn.net/liuweiyuxiang/article/details/78220080
        # y_test = y_test.flatten()
        # print("y_test", y_test)
        y_training = X_training[:, -5:]
        # y_training = y_training.flatten()
        y_train = y_training.astype(np.float)
        y_test = y_test.astype(np.float)
        X_test = X_test[:, :-5]
        X_training = X_training[:, :-5]
        X_test = X_test.astype(np.float)
        X_training = X_training.astype(np.float)
        test_attri_list.append(X_test)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training)
        training_class_names_list.append(y_train)
        # print("start is",start)
        # print("end is",end)
        start = end
        end = end + num_test
    # print("training_class_names_list", training_class_names_list)
    return Feature_name, test_attri_list, test_class_names_list, training_attri_list, training_class_names_list  # (X_test,y_test,X_training,y_train)


# To calculate Tao_value , the input value should be transport by the get_remainder_dict function
def taoob (attri_list):
    # add a function tao !!
    # print(np.array(attri_list).shape)
    Treat_acc = 0
    number_sample = len(attri_list)
    if number_sample == 0:
        # print('THIS IS SOME WRONG!!!')
        return 0
    sum0 = 0
    sum1 = 0
    tot0 = 0
    tot1 = 0
    for item in attri_list:
        label = item[1]
        # print(label)
        Y = label[1]
        '''
        if label[4]>0.95:
            label[4] = 0.95
        elif label[4] < 0.05:
            label[4] = 0.05
        '''
        if label[0]:
            # print(label[1], label[4])
            sum1 += Y / label[4]
            tot1 += 1 / label[4]
        else:
            # print(label[1], 1 - label[4])
            sum0 += Y / (1 - label[4])
            tot0 += 1/(1 - label[4])
    '''
    print('sum0 = ', sum0)
    print('sum1 = ', sum1)
    print('tot0 = ', tot0)
    print('tot1 = ', tot1)
    '''
    if tot1 == 0:
        # Acc = 0
        return 0
    else:
        Acc = sum1/tot1
    if tot0 == 0:
        # Ref = 0
        return 0
    else:
        Ref = sum0/tot0
    taoob = Acc - Ref
    # print('taoob : ', taoob)
    tao = sum(taoob)
    # print("tao : ", tao)
    return tao


def Prework(attri_list):
    temptdata = attri_list[0]
    # print("input data to get tao = ", temptdata)
    Tao_Value = taoob(temptdata)
    # print('Tao_initial = ', Tao_Value)
    return Tao_Value


def build_dict_of_attributes_with_class_values(X, y):  # ,feature):
    dict_of_attri_class_values = {}
    fea_list = []
    for i in range(X.shape[1]):  #  map all features
        fea = i
        l = X[:, i]
        # print(l)
        attribute_list = []
        count = 0
        # add all features to the dic and add the labal belong to every sample
        for j in l:
            # print('j: ', j)
            attribute_value = []
            attribute_value.append(j)
            attribute_value.append(y[count, :])
            attribute_list.append(attribute_value)
            count += 1
        dict_of_attri_class_values[fea] = attribute_list
        fea_list.append(fea)
    # print("dict_of_attri_class_values: ", dict_of_attri_class_values[0][1])
    return dict_of_attri_class_values, fea_list
    # features_with_max_gain_and_theta(dict_of_attri_class_values)


class Node(object):
    def __init__(self, val, lchild, rchild, feature, FeatureNumber, the, depth, leaf, Prune_value):
        self.root_value = val
        self.root_left = lchild
        self.root_right = rchild
        self.feature = feature
        self.theta = the
        self. Prune_value = Prune_value
        self.depth = depth
        self.leaf = leaf ### bool type
        self.FeatureNumber = FeatureNumber

    # method to identify if the node is leaf
    def is_leaf(self):
        return self.leaf

    # method to return threshold value
    def ret_thetha(self):
        return self.theta

    def ret_root_value(self):
        return self.root_value

    def ret_llist(self):
        return self.root_left

    def ret_rlist(self):
        return self.root_right

    def ret_depth(self):
        return self.depth


    def ret_feature(self):
        return self.feature


    def __repr__(self):
        return "(%r, %s, %r, %r, %r, %r, %r)" % (self.root_value, self.feature, self.theta, self.leaf, self.depth, self.root_left, self.root_right,)


# Decision Tree object
class DecisionTree(object):
    fea_list = []

    def __init__(self):
        self.root_node = None

    # fit the decision tree
    def fit(self, dict_of_everything, cl_val, eta_min_val, Feature_name, Tao):
        root_node = self.create_decision_tree(dict_of_everything, cl_val, eta_min_val, Feature_name, Tao, 1)  # ,fea_list)
        return root_node

    # calculate the mean values for all the class labels
    def cal_mean_class_values(self, class_values):
        mean_val = sum(class_values) / float(len(class_values))
        # print(mean_val)
        return mean_val

    # method to calculate best threshold value for each feature
    def cal_best_theta_value(self, ke, attri_list):  ### The atri_listt is the value belong to one feature [1] is the label
        data = []
        # class_values = []
        # print("ke is: ", ke)
        # print(attri_list, ': is attri_list')
        for i in attri_list:
            # val = float(i[0])
            data.append(i[0])
            # class_values.append(i[1])
        # print('Classlabel: ', class_values)
        # print('data: ', data)
        ## We should calculate the tao value here not MSE
        # mse_parent = mean_sqaured_error(class_values)
        # print("mse for parent",mse_parent)
        # print("Entropy of parrent",entropy_of_par_attr)
        max_tao = 0
        tao_child = []
        theta = 0
        best_index_left_list = []
        best_index_right_list = []
        class_labels_list_after_split = []
        # print(data)
        # data = list(data)
        data.sort()
        data = set(data)
        data = list(data)
        # print('Data: ', data)
        for i in range(len(data) - 1):
            cur_theta = float(float(data[i]) + float(data[i + 1])) / 2
            # print("cur thetha",cur_theta)
            # print(data[i] +"ji"+ data[i+1],cur_theta)
            index_less_than_theta_list = []
            values_less_than_theta_list = []
            index_greater_than_theta_list = []
            values_greater_than_theta_list = []
            count = 0
            ### enumerate : both index and value:https://blog.csdn.net/liu_xzhen/article/details/79564455
            for c, j in enumerate(attri_list):
                # print(c, 'J is : ', j)
                if j[0] <= cur_theta:
                    # print("J[0] less", j[0])
                    values_less_than_theta_list.append(j)
                    index_less_than_theta_list.append(c)
                else:
                    # print("J[0] grater",j[0])
                    values_greater_than_theta_list.append(j)
                    index_greater_than_theta_list.append(c)
                # count += 1
            # print('values_greater_than_theta_list: ', values_greater_than_theta_list)
            # print("Len og less list",len(index_less_than_theta_list))
            # print("len og greater list",len(index_greater_than_theta_list))
            tao_left = taoob(values_less_than_theta_list)
            # print(entropy_of_less_attribute)
            tao_right = taoob(values_greater_than_theta_list)
            ## we use sum sqaure here
            tao_split = tao_left ** 2 + tao_right ** 2
            if tao_split > max_tao:
                max_tao = tao_split
                tao_child = [tao_left, tao_right]
                theta = cur_theta
                best_index_left_list = index_less_than_theta_list
                best_index_right_list = index_greater_than_theta_list
                class_labels_list_after_split = values_less_than_theta_list + values_greater_than_theta_list
        # print('split left: ', best_index_left_list)
        # print('split right: ', best_index_right_list)
        return max_tao, theta, best_index_left_list, best_index_right_list, class_labels_list_after_split, tao_child

    # method to select the best feature out of all the features.
    ### the dict_rep is
    def best_feature(self, dict_rep):
        # dict_theta = {}
        # dict_theta = {}
        key_value = None
        best_tao_split = -1
        best_theta = 0
        best_index_left_list = []
        best_index_right_list = []
        # best_mse_left = -1
        # best_mse_right = -1
        best_class_labels_after_split = []
        tmp_list = []
        best_tao_child = []
        for ke in dict_rep.keys():
            # print("Key now is", ke, 'dict_rep is ', dict_rep[ke])
            tao_split, theta, index_left_list, index_right_list, class_labels_after_split, tao_child = self.cal_best_theta_value(ke, dict_rep[ke])
            # print("Best theta is", ke,info_gain,theta,index_left_list)#,index_right_list)
            if tao_split > best_tao_split:
                best_tao_split = tao_split
                best_tao_child = tao_child
                best_theta = theta
                key_value = ke
                best_index_left_list = index_left_list
                best_index_right_list = index_right_list
                best_class_labels_after_split = class_labels_after_split
        tmp_list.append(key_value)
        # tmp_list.append(best_info_gain)
        tmp_list.append(best_theta)
        tmp_list.append(best_index_left_list)
        tmp_list.append(best_index_right_list)
        tmp_list.append(best_class_labels_after_split)
        tmp_list.append(best_tao_child)
        return tmp_list

    def get_remainder_dict(self, dict_of_everything, index_split):
        # global fea_list
        splited_dict = {}
        for ke in dict_of_everything.keys():
            val_list = []
            modified_list = []
            l = dict_of_everything[ke]
            # print(ke,index_left_split)
            # print(l)
            for i, v in enumerate(l):
                # print(i,v)
                if i not in index_split:
                    # print(ke,i,v)
                    modified_list.append(v)
                    val_list.append(v[1])
            # print(modified_list)
            splited_dict[ke] = modified_list
        return splited_dict, val_list

    # method to create decision tree
    def create_decision_tree(self, dict_of_everything, class_val, eta_min_val, Featurename, Tao, depth):  # ,fea_list):
        if len(class_val) < eta_min_val:   ### if the number of the leaf < the minest number we having setting
            # majority_val = self.cal_mean_class_values(class_val)
            # print("Leaf node for less than 8 is",majority_val, len(class_val))#,class_val)
            root_node = Node(Tao, None, None, None, None, None, depth, True, None)
            # print('Node number: ', len(class_val))
            return root_node
        else:
            best_features_list = self.best_feature(dict_of_everything)
            # print(best_features_list)
            node_name = best_features_list[0]
            theta = best_features_list[1]
            index_left_split = best_features_list[2]
            # print("Length of left split",len(index_left_split))#,index_left_split)
            index_right_split = best_features_list[3]
            # print("Length of right split",len(index_right_split))#,index_right_split)
            ## use tao to replace the class_values
            class_values = best_features_list[4]
            Taovalue = best_features_list[5]
            # print ("Length of class values", len(class_values))
            left_dict, class_val1 = self.get_remainder_dict(dict_of_everything, index_left_split)
            # print("index of left split",len(index_left_split))
            # print("Left class values is",len(class_val1))
            right_dict, class_val2 = self.get_remainder_dict(dict_of_everything, index_right_split)
            # print("indx of right split",len(index_right_split))
            # print("right class values is",len(class_val2))
            ##Add the tao value of each child note here!!!
            leftchild = self.create_decision_tree(left_dict, class_val1, eta_min_val, Featurename, Taovalue[0], depth+1)
            # leftchild = None
            rightchild = self.create_decision_tree(right_dict, class_val2, eta_min_val, Featurename, Taovalue[1], depth+1)
            root_node = Node(Tao, leftchild, rightchild, Featurename[node_name], node_name, theta, depth, False, None)
            return root_node

    # method to predict the values for test data
    def predict(self, X, root):
        predicted_list = []
        for row in X:
            y_pred = self.classify(row, root)
            predicted_list.append(y_pred)
        return predicted_list

    def classify(self, row, root):
        dict_test = {}
        for k, j in enumerate(row):
            dict_test[k] = j
        # print(dict_test)
        current_node = root
        while not current_node.leaf:
            if dict_test[current_node.root_value] <= current_node.theta:
                current_node = current_node.root_left
            else:
                current_node = current_node.root_right
        # print(current_node.root_value,dict_test[current_node.root_value], current_node.theta)
        return current_node.root_value


'''
def CountChildren (Tree):
    Num_leaf = 0
    Num_node = 0
    if Tree.leaf:
        # print('Here ',Tree.feature)
        Num_leaf += 1
    else:
        Num_node += 1
        LeftTree = Tree.root_left
        # print('left Tree', LeftTree.root_left)
        # print('Tree leaf ', LeftTree.leaf)
        RightTree = Tree.root_right
        Add_node, Add_leaf = CountChildren(LeftTree)
        Num_leaf += Add_leaf
        Num_node += Add_node
        Add_node, Add_leaf = CountChildren(RightTree)
        Num_leaf += Add_leaf
        Num_node += Add_node
    # print('Num :', Num_node)
    return Num_node, Num_leaf
'''


def CountChildren (Tree):
    Num_node = 0
    if Tree.root_left == None and Tree.root_right == None:
        # print('Here ',Tree.root_left, Tree.root_right)
        Num_node += 1
    else:
        # print('There ',Tree.root_left, Tree.root_right)
        LeftTree = Tree.root_left
        # print('left Tree', LeftTree.root_left)
        # print('Tree leaf ', LeftTree.leaf)
        RightTree = Tree.root_right
        Num_node += CountChildren(LeftTree)
        Num_node += CountChildren(RightTree)
    # print('Num :', Num_node)
    return Num_node


def Taocv(Label):  # Edit
    # print('label: ', len(Label))
    number_sample = len(Label)
    if number_sample == 0:
        # print('THIS IS SOME WRONG!!!')
        return 0
    sum0 = 0
    sum1 = 0
    tot0 = 0
    tot1 = 0
    for item in Label:
        label = item
        # print('666 ', label[0])
        Y = label[1]
        if label[0]:
            # print(label[1], label[4])
            sum1 += Y / label[4]
            tot1 += 1 / label[4]
        else:
            # print(label[1], 1 - label[4])
            sum0 += Y / (1 - label[4])
            tot0 += 1 / (1 - label[4])
    '''
    print('sum0 = ', sum0)
    print('sum1 = ', sum1)
    print('tot0 = ', tot0)
    print('tot1 = ', tot1)
    '''
    if tot1 == 0:
        Acc = 0
    else:
        Acc = sum1 / tot1
    if tot0 == 0:
        Ref = 0
    else:
        Ref = sum0 / tot0
    taoob = Acc - Ref
    # print('taoob : ', taoob)
    tao = sum(taoob)
    # print("tao : ", tao)
    return tao


# calculate the number of leafs in these suntrees and the Q_prune of the tree and subtrees
def CountNodes(Tree):  # EDIT
    if Tree.leaf:
        return 1, None, None, Tree.depth, Tree.root_value ** 2
    Node_left, Q_left_min, dic_left_prune, l_depth, left_leaf_value = CountNodes(Tree.root_left)
    Node_righ, Q_righ_min, dic_righ_prune, r_depth, righ_leaf_value = CountNodes(Tree.root_right)
    Node_number = Node_left + Node_righ
    leaf_value = left_leaf_value + righ_leaf_value
    dic_Q_prune = {}
    Q_min = 3333333
    if Q_left_min and Q_righ_min:
        if Q_left_min < Q_righ_min:
            dic_Q_prune = dic_left_prune
            Q_min = Q_left_min
            depth = l_depth
        else:
            dic_Q_prune = dic_righ_prune
            Q_min = Q_righ_min
            depth = r_depth
    elif Q_left_min:
        dic_Q_prune = dic_left_prune
        Q_min = Q_left_min
        depth = l_depth
    elif Q_righ_min:
        dic_Q_prune = dic_righ_prune
        Q_min = Q_righ_min
        depth = r_depth

    Q_prune = (leaf_value / Node_number - Tree.root_value ** 2) / (Node_number - 1)
    Tree.Prune_value = Q_prune
    if Q_prune < Q_min:
        Q_min = Q_prune
        dic_Q_prune = Tree
        depth = Tree.depth

    return Node_number, Q_min, dic_Q_prune, depth, leaf_value


def FindSubTree(Tree, Q, depth):
    SearchEnding = 0
    if Tree.leaf:
        return Tree, 0
    if Tree.Prune_value == Q and Tree.depth == depth:
        # print('Merge')
        Tree.root_left = None
        Tree.root_right = None
        Tree.leaf = True
        return Tree, 1
    Tree.root_left, SearchEnding = FindSubTree(Tree.root_left, Q, depth)
    if SearchEnding:
        return Tree, 1
    Tree.root_right, SearchEnding = FindSubTree(Tree.root_right, Q, depth)
    if SearchEnding:
        return Tree, 1
    return Tree, 0


# to get the Q_prune value
def Find_least_Qp(Tree):  # edit
    Node_number, Q_min, dic_Q_prune, depth, leaf_value = CountNodes(Tree)
    subTree, search = FindSubTree(Tree, Q_min, depth)
    alpha = Q_min
    return subTree, alpha


# use test data to get MSE
def Get_MSE(Tree, X_test, Y_test, MSECV, MSETR):  # edit
    if Tree.root_left == None and Tree.root_right == None:
        taocv = Taocv(Y_test)  # you should change the form of the input data X_test
        MSECV += taocv * Tree.root_value
        MSETR += Tree.root_value ** 2
        return MSECV, MSETR
    else:
        X_left = []
        Y_left = []
        X_right = []
        Y_right = []
        Comp = Tree.theta
        key = Tree.FeatureNumber
        #print('Comp: ', Comp, 'Key: ', key)
        # print(X_test[0][key])
        for item, subset in enumerate(X_test):
            temp = subset
            if temp[key] < Comp:
                X_left.append(temp)
                Y_left.append(Y_test[item])
            else:
                X_right.append(temp)
                Y_right.append(Y_test[item])
        if len(Y_left) > 0:
            print('lY: ', len(Y_left))
            dCV, dTR = Get_MSE(Tree.root_left, X_left, Y_left, MSECV, MSETR)
            MSECV += dCV
            MSETR += dTR
        if len(Y_right) > 0:
            print('rY: ', len(Y_right))
            print('leaf: ', Tree.root_right.leaf)
            dCV, dTR = Get_MSE(Tree.root_right, X_right, Y_right, MSECV, MSETR)
            MSECV += dCV
            MSETR += dTR
        return MSECV, MSETR


def SetCross(Subtree, X_test, Y_test):
    Number = CountChildren(Subtree)
    print("the child node: ", Number)
    MSECV, MSETR = Get_MSE(Subtree, X_test, Y_test, 0, 0)
    print('MSECV: ', MSECV, 'MSETR: ', MSETR)
    Number_test = len(X_test)
    # print("Number of test data:", Number_test)
    MSE_tao = -2 / Number_test * MSECV + MSETR / Number_test
    print('MSE_tao :', MSE_tao)
    return MSE_tao


# prune in crossdata a
def CrossPprune(Tree, X_test, Y_test):  # edit
    from copy import deepcopy
    # print('Y len :', len(Y_test))
    # print('Y:', Y_test[1])
    Tree_now = deepcopy(Tree)
    Nodes = []
    Paralpha = []
    Nodes.append(deepcopy(Tree_now))
    Paralpha.append(SetCross(deepcopy(Tree_now), X_test, Y_test))
    while 1:
        if Tree_now.root_left == None and Tree_now.root_right == None:
            # if Tree_now.leaf:
            # print('break tree:', Tree_now)
            break
        Tree_now, alpha = Find_least_Qp(Tree_now)  # remember let tree.leaf = True after merge
        # print('Type: ', type(Tree_now))
        Nodes.append(deepcopy(Tree_now))
        Paralpha.append(alpha)
    print('Nodes: ', Nodes)
    print("len node: ", len(Nodes))
    print("Alpha: ", Paralpha)
    MSQ_min = 2147483648  # a small number -2147483648
    AnsList = []
    best_node = Tree
    best_alpha = None
    for item, sub in enumerate(Nodes):
        # print('each item :', sub)
        MSQ = SetCross(sub, X_test, Y_test)
        if MSQ <= MSQ_min:
            MSQ_min = MSQ
            best_node = sub
            best_alpha = Paralpha[item]
    AnsList.append(MSQ_min)
    AnsList.append(best_node)
    AnsList.append(best_alpha)
    return AnsList
'''
def prune(Tree, alpha, k, eta_min):
    dif = 0
    rdif = 0
    LeftTree = Tree.root_left
    RightTree = Tree.root_right
    # pass
    if LeftTree.leaf == 0:
        LeftTree, ldif = prune(LeftTree, alpha, k, eta_min)
        dif += ldif
        k -= ldif
    if RightTree.leaf == 0:
        RightTree, rdif = prune(RightTree, alpha, k, eta_min)
        dif += rdif
        k -= rdif
    # if there still have child tree, which meaning that parent don't need to prune
    if not LeftTree.leaf or not RightTree.leaf:
        Tree.root_left = LeftTree
        Tree.root_right = RightTree
        return Tree, dif
    else:  # decision if need prune
        Q_prune_child = LeftTree.root_value ** 2 + RightTree.root_value ** 2 - alpha * k
        Q_prune_parent = Tree.root_value ** 2 - alpha * (k - 1)
        if (Q_prune_child - Q_prune_parent) < eta_min:
            print('Prune happened! : ')
            Tree.leaf = True
            Tree.root_left = None
            Tree.root_right = None
            dif += 1
            # print('k: ', k)
    return Tree, dif
'''


def main(num_arr, eta_min, Pi):
    eta_min_val = round(eta_min * num_arr.shape[0])
    print('eta_min_val : ', eta_min_val)
    # randomly shuffle the array so that we can divide the data into test/training
    # random_arr1 = random_numpy_array(num_arr)
    # divide data into test labels,test features,training labels, training features
    Feature_name, test_attri_list, test_class_names_list, training_attri_list, training_class_names_list = generate_set(num_arr, Pi, CrossTest=1)
    accu_count = 0
    test_fin_mse = 0
    pred_fin = 0
    # ten fold iteration for each eta-min value
    for i in range(1):
        # build a dictionary with class labels and respective features values belonging to that class
        dict_of_input, fea = build_dict_of_attributes_with_class_values(training_attri_list[i], training_class_names_list[i])
        # print(dict_of_input)
        # instantiate decision tree instance
        build_dict = DecisionTree()
        Tao_initial = Prework(dict_of_input)
        # build the decision tree model.
        dec = build_dict.fit(dict_of_input, training_class_names_list[i], eta_min_val, Feature_name, Tao_initial)
        # predict the class labels for test features
        '''
        l = build_dict.predict(test_attri_list[i], dec)
        # calculate the mean squared error measure for predicited test data
        mse = accuracy_for_predicted_values(test_class_names_list[i], l)
        # print("Number of right values are",right,"Wrong ones are",wrong)
        # accu_count += accu
        test_fin_mse += mse
        # pred_fin += pred
    print("Average MSE for eta min of", eta_min, "is", float(test_fin_mse) / 10) 
    '''
        Num_node = CountChildren(dec)
        print('Num of node is ', Num_node)
        print('The original tree: ', dec)
    print('Y: ', len(test_class_names_list[0]))
    AnsList = CrossPprune(dec, test_attri_list[0], test_class_names_list[0])
    MSQ = AnsList[0]
    best_dec = AnsList[1]
    best_alpha = AnsList[2]
    print('The value of alpha: ', best_alpha)
    Num_node = CountChildren(best_dec)
    print('After pruning, num of node is ', Num_node)
    print("Average MSE for eta min of ", MSQ)
    print("Tree: ", best_dec)


if __name__ == "__main__":
    '''if len(sys.argv) == 2:    ### sys.argv ????  what's mean ? is any wrong here?
        newfile = sys.argv[1]
        # load the data file and do the preprocessing
        num_arr = load_csv(newfile)
        # for each threshold value run the classifier for 10 cross-validation
        eta_min_list = [0.05, 0.10, 0.15, 0.20]
        for i in eta_min_list:
            main(num_arr, i)'''
    newfile = 'gbsg.csv'
    # load the data file and do the preprocessing
    Pi = lr.CaculatePi(newfile)
    num_arr = load_csv(newfile)
    # for each threshold value run the classifier for 10 cross-validation
    eta_min_list = [0.15, 0.20, 0.25, 0.30]
    main(num_arr, 0.05, Pi)

    #for i in eta_min_list:
     #   main(num_arr, i, Pi)
