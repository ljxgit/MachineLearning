# -*- coding:utf-8 -*-
#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################
from LR import *

def loadData():
    train_data = []
    train_label = []
    with open('F:\My_Program\Python_program\Machine Learning\Logistic Regression\dataset.txt') as fileIn:
        for line in fileIn.readlines(): # readlines读取多行，readline每次读取一行
            lineArr = line.strip().split() # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
            train_data.append([1.0, float(lineArr[0]), float(lineArr[1])]) # x0=1，第一维特征为1
            train_label.append(float(lineArr[2]))
    return mat(train_data), mat(train_label).transpose() # 100*3,1*100，将一个list转成matrix

## step 1: load data
# def loadData():
#     train_data = []
#     train_label = []
#     fileIn = open('F:/My_Program/Python_program/V1_RF_training/chair_feature_data.txt')
#     a = fileIn.readlines()
#     for line in a: # readlines读取多行，readline每次读取一行
#         lineArr = line.strip('[]').split(',') # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
#         train_data.append([float(lineArr[0]), float(lineArr[1])]) # x0=1，第一维特征为1
#         train_label.append(float(lineArr[2]))
#     fileIn.close()
#     return mat(train_data), mat(train_label).transpose() # 100*3,1*100，将一个list转成matrix

print("step 1: load data...")
train_data, train_label = loadData()
test_data = train_data
test_label = train_label

## step 2: training...
print("step 2: training...")
opts = {'alpha': 0.01, 'maxIter': 200, 'optimizeType': 'smoothStocGradDescent'} # 参数选择，学习率、最大迭代次数、优化算法类型
optimalWeights = trainLogRegres(train_data, train_label, opts)

## step 3: testing
print("step 3: testing...")
accuracy = testLogRegres(optimalWeights, test_data, test_label)
## step 4: show the result
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showLogRegres(optimalWeights, train_data, train_label)