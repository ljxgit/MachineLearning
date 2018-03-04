# -*- coding:utf-8 -*-
#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import time
import cv2
import matplotlib.pyplot as plt
# a=cv2.imread('F:\My_Program\Python_program\V2_shape\edge_test.png',0)
# b=plt.imshow(a)
# plt.colorbar(b)
# plt.show()

def calcKernelValue(train_data, sample_data, kernelOption):  # 计算每一个样本Xi与所有样本的核内积
    kernelType = kernelOption[0]
    numSamples = train_data.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    if kernelType == 'linear':
        kernelValue = train_data * sample_data.T
    elif kernelType == 'rbf': # 径向基核函数
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(numSamples):
            diff = train_data[i, :] - sample_data
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue


# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_data, kernelOption):  # 所有的样本与其他样本的核内积，80*80矩阵
    numSamples = train_data.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_data, train_data[i, :], kernelOption)
    return kernelMatrix


# define a struct just for storing variables and data
class SVMStruct:  # SVM相关的参数
    def __init__(self, dataSet, labels, C, toler, kernelOption): # 类的构造函数
        self.train_data = dataSet  # each row stands for a sample
        self.train_label = labels  # corresponding label
        self.C = C  # slack variable，惩罚系数，C越大，间隔越小，泛化能力越低
        self.toler = toler  # termination condition for iteration 迭代终止条件
        self.numSamples = dataSet.shape[0]  # number of samples
        self.alphas = mat(zeros((self.numSamples, 1)))  # Lagrange factors for all samples
        self.b = 0  # 偏移量
        self.errorCache = mat(zeros((self.numSamples, 2))) # 第二列作为更新后的误差
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_data, self.kernelOpt)  # 核矩阵


# calculate the error for alpha k，SVM系数 某个样本对应的alpha k
def calcError(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_label).T * svm.kernelMat[:, alpha_k] + svm.b)  # 核内积矩阵
    error_k = output_k - float(svm.train_label[alpha_k])  # 计算误差
    return error_k


# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]


# select alpha j which has the biggest step，|Ei-Ej|最大是选取alpha j 的标准
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]  # mark as valid(has been optimized)
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0]  # mat.A return array
    maxStep = 0
    alpha_j = 0
    error_j = 0

    # find the alpha with max iterative step
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep: # 保存最大误差，用于比较
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    # if came in this loop first time, we select alpha j randomly
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.numSamples))
        error_j = calcError(svm, alpha_j)

    return alpha_j, error_j


# the inner loop for optimizing alpha i and alpha j
def innerLoop(svm, alpha_i): #  先选alpha i，再在不违反KKT条件的乘子中，选alpha j
    error_i = calcError(svm, alpha_i)
    ### check and pick up the alpha who violates the KKT condition，选取最违反KKT条件的作为alpha i
    ## satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    # 判断第i个系数alpha_i是否违反KKT条件, toler是满足KKT条件的容忍值
    if (svm.train_label[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or \
                    (svm.train_label[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # step 1: select alpha j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if svm.train_label[alpha_i] != svm.train_label[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        # step 3: calculate eta (the similarity of sample i and j) 衡量样本i、j的相似性
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
              - svm.kernelMat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # step 4: update alpha j
        svm.alphas[alpha_j] -= svm.train_label[alpha_j] * (error_i - error_j) / eta

        # step 5: clip alpha j 按照上下边界裁剪alpha_j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # step 6: if alpha j not moving enough, just return
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j) # 每做完一次优化，就要更新一次误差
            return 0

        # step 7: update alpha i after optimizing aipha j
        svm.alphas[alpha_i] += svm.train_label[alpha_i] * svm.train_label[alpha_j] \
                               * (alpha_j_old - svm.alphas[alpha_j])

        # step 8: update threshold b
        b1 = svm.b - error_i - svm.train_label[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_i] \
             - svm.train_label[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_label[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_j] \
             - svm.train_label[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # step 9: update error cache for alpha i, j after optimize alpha i, j and b
        # 每次优化完i、j、b后，都要更新误差
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0


# the main training procedure
def trainSVM(train_data, train_label, C, toler, maxIter, kernelOption=('rbf', 1.0)):
    # calculate training time
    startTime = time.time()

    # init data struct for svm，toler是迭代终止条件，即KKT条件的容忍度
    svm = SVMStruct(mat(train_data), mat(train_label), C, toler, kernelOption)

    # start training
    entireSet = True
    alphaPairsChanged = 0 # alpha系数的改变量
    iterCount = 0 # 迭代次数
    # Iteration termination condition:
    # 	Condition 1: reach max iteration
    # 	Condition 2: no alpha changed after going through all samples,
    # 				 in other words, all alpha (samples) fit KKT condition
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0

        # update alphas over all training examples
        if entireSet:
            for i in range(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i) # 如果有违反KKT条件的系数，innerloop返回1，i即alpha_i，在loop内确定第二个参数alpha_j
            print('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1
        # update alphas over examples where alpha is not 0 & not C (not on boundary)，只更新支持向量的系数
        else:
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)
            print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1

        # alternate loop over all examples and non-boundary examples
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    return svm # 将svm的若干参数返回，包括训练得到的svm.alphas


# testing your trained svm model given test set
def testSVM(svm, test_data, test_label):
    test_data = mat(test_data)
    test_label = mat(test_label)
    numTestSamples = test_data.shape[0]
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0] # 返回系数大于0的下标，即支持向量的下标
    supportVectors = svm.train_data[supportVectorsIndex] # 找出所有的支持向量
    supportVectorLabels = svm.train_label[supportVectorsIndex] # 支持向量对应的标签
    supportVectorAlphas = svm.alphas[supportVectorsIndex] # 支持向量对应的系数
    matchCount = 0
    for i in range(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, test_data[i, :], svm.kernelOpt)
        predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        if sign(predict) == sign(test_label[i]):
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    return accuracy


# show your trained svm model only available with 2-D data
def showSVM(svm):
    if svm.train_data.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # draw all samples
    for i in range(svm.numSamples):
        if svm.train_label[i] == -1:
            plt.plot(svm.train_data[i, 0], svm.train_data[i, 1], 'or')
        elif svm.train_label[i] == 1:
            plt.plot(svm.train_data[i, 0], svm.train_data[i, 1], 'ob')

    # mark support vectors
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.train_data[i, 0], svm.train_data[i, 1], 'oy')

    # draw the classify line
    w = zeros((2, 1))
    for i in supportVectorsIndex:
        w += multiply(svm.alphas[i] * svm.train_label[i], svm.train_data[i, :].T)
    min_x = min(svm.train_data[:, 0])[0,0]
    max_x = max(svm.train_data[:, 0])[0, 0]
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.title('SVM Classifier')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()