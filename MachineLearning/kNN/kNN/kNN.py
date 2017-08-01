# coding: gbk

import numpy as np
import operator
import os
import cv2

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = list("AABB")

    return group, labels

def classify0(inX, dataSet, labels, k):
    """ k-近邻算法 """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    distances = np.sum(diffMat**2, axis=1)**0.5
    sortedDistIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def file2matrix(filename):
    """ 将文本记录转换为Numpy数组的解析程序 """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = np.empty((numberOfLines,), np.int)
    index = 0
    for index, line in enumerate(arrayOLines):
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector[index] = int(listFromLine[-1])

    fr.close()
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """ 归一化特征值 """
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet /= np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

def datingClassTest(hoRatio = 0.10):
    """ 分类器针对约会网站的测试代码 """
    datingDataMat, datingLabels = file2matrix('.data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    print("The classifier    The real answer")
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], 
                                     datingLabels[numTestVecs:m], 3)
        print("            %d    %d" % (classifierResult, datingLabels[i]), end="")
        if (classifierResult != datingLabels[i]):
            errorCount += 1
            print("  *")
        else:
            print("")
    print("The total error rate is: %.2f%%" % (errorCount/numTestVecs*100))

    return

def classifyPerson():
    """ 约会网站测试函数 """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("Percentage of time spent playing video games? "))
    ffMiles = float(input("Frequent flier miles earned per year? "))
    iceCream = float(input("Liters of ice cream consumed per year? "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

    return


def img2vector(file):
    """ txt文件转向量 """
    returnVect = np.zeros((1, 1024))
    fr = open(file)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()

    return returnVect


def handwritingClassTraining():
    """ 手写数字训练矩阵 """
    hwLabels = []
    trainingFileList = os.listdir("./data/digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("./data/digits/trainingDigits/%s" % fileNameStr)

    return trainingMat, hwLabels


def handwritingTest():
    """ 手写数字分类器测试 """
    testFileList = os.listdir("./data/digits/testDigits")
    errorCount = 0
    mTest = len(testFileList)

    trainingMat, hwLabels = handwritingClassTraining()

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("./data/digits/testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("%6s  %d  %d" % (fileNameStr[:-4], classifierResult, classNumStr), end='')

        if classifierResult != classNumStr:
            errorCount += 1
            print("  *")
        else:
            print("")
            
    print("\nThe total number of errors is: %d" % errorCount)
    print("\nThe total error rate is: %.2f%%" % (errorCount/mTest*100))
    
    return


def png2vec(filename):
    """ 将png图片转为向量 """
    # 1. 读取图片, 裁剪
    img = cv2.imread(filename)
    if img is None:
        print("Read image error!!!")
        exit()
    ret, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    biImg = cv2.split(thresh)[0]    # 取出红色通道(反正图片只有黑白色)

    # 2. 转为 32*32 矩阵
    left, right, top, bottom = 0, 0, 0, 0
    flag = True
    for i, row in enumerate(biImg):
        if flag and np.unique(row).shape[0] > 1:
            top = i
            flag = False
            continue
        if not flag and np.unique(row).shape[0] > 1:
            bottom = i
    
    flag = True
    for i, col in enumerate(biImg.T):
        if flag and np.unique(col).shape[0] > 1:
            left = i
            flag = False
            continue
        if not flag and np.unique(col).shape[0] > 1:
            right = i
    
    height = bottom + 1 - top
    width = right + 1 - left
    if height > width:
        delta = (height - width) // 2
    else:
        delta = 0
    img = biImg[top:bottom+1, left-delta:right+1+delta]
    img = 1 - cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    filepath = "./data/digits/Mydigits/" + filename[-5] + ".txt"
    np.savetxt(filepath, img, fmt='%d', delimiter='', newline='\r\n')

    return filepath
