# -*- coding:utf-8 -*-
#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import matplotlib.pyplot as plt
import time

# 注意！！！矩阵计算后结果仍然是一个矩阵，1*3和3*1矩阵相乘后是一个1*1的矩阵

# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# train a logistic regression model using some optional optimize algorithm
# input: train_data is a mat datatype, each row stands for one sample
# train_label is mat datatype too, each row is the corresponding label
# pts is optimize option include step(learning rate) and maximum number of iterations 迭代最大次数
def trainLogRegres(train_data, train_label, opts): # opts是一个字典类型
    # calculate training time
    startTime = time.time()
    numSamples, numFeatures = shape(train_data)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = ones((numFeatures, 1)) # 权值初始化为1，矩阵格式

    # optimize through gradient descent algorilthm
    for k in range(maxIter): # 最大迭代次数
        if opts['optimizeType'] == 'GradDescent':  # gradient descent algorilthm
            output = sigmoid(train_data * weights) # 批量梯度下降，每次迭代计算所有样本梯度
            error = output - train_label # error是对数损失函数向量化表示后的一部分，便于用向量矩阵的形式计算参数更新
            weights = weights - alpha * train_data.transpose() * error
        elif opts['optimizeType'] == 'StocGradDescent':  # stochastic gradient descent
            for i in range(numSamples):
                output = sigmoid(train_data[i, :] * weights) # 随机梯度下降，每次迭代选取一个样本，计算梯度，“在线学习”、增量式学习
                # SGD每新来一个样本，就可以实时地更新权值，在线学习
                error = train_label[i, 0] - output
                weights = weights + alpha * train_data[i, :].transpose() * error
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = list(range(numSamples))
            for i in range(numSamples): # 改进的随机梯度下降，每次随机挑选一个样本来更新系数，且学习率逐步下降
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex))) # 随机挑选一个样本
                output = sigmoid(train_data[randIndex, :] * weights)
                error = train_label[randIndex, 0] - output
                weights = weights + alpha * train_data[randIndex, :].transpose() * error
                del dataIndex[randIndex]  # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')

    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    return weights  # 返回训练好的系数


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_data, test_label):
    numSamples, numFeatures = shape(test_data)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid((test_data[i, :] * weights)[0,0]) > 0.5
        if predict == bool(test_label[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_data, train_label):
    # notice: train_data and train_label is mat datatype
    numSamples, numFeatures = shape(train_data)
    if numFeatures != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # draw all samples，不同类不同颜色
    for i in range(numSamples):
        if int(train_label[i, 0]) == 0:
            plt.plot(train_data[i, 1], train_data[i, 2], 'or')
        elif int(train_label[i, 0]) == 1:
            plt.plot(train_data[i, 1], train_data[i, 2], 'ob')

    # draw the classify line
    min_x = min(train_data[:, 1])[0, 0]
    max_x = max(train_data[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Logistic Regression')
    plt.show()