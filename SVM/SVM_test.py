# -*- coding:utf-8 -*-
#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import SVM

################## test svm #####################
## step 1: load data
print("step 1: load data...")
dataSet = []
labels = []
fileIn = open('F:\My_Program\Python_program\Machine Learning\SVM\dataset.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])]) # Wx+b，不用设x0=1
	labels.append(float(lineArr[2]))

dataSet = mat(dataSet)
labels = mat(labels).T
train_data = dataSet[0:79, :]
train_label = labels[0:79, :]
test_data = dataSet[80:99, :]
test_label = labels[80:99, :]

## step 2: training...
print("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_data, train_label, C, toler, maxIter, kernelOption = ('linear', 0))

## step 3: testing
print("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_data, test_label)

## step 4: show the result
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
SVM.showSVM(svmClassifier)