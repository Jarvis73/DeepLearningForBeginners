################################################
#   coding: gbk
#
#   对数几率回归 ( logistic regression ): 
#  
#   《机器学习》 周志华 清华大学出版社
#
#   第3章 线性模型 习题3.3
#
#   目标: 实现对数几率回归, 并给出西瓜数据集 3.0 alpha 上的结果
#
#   Writen by Jarvis (zjw.math@qq.com)
#
#   Date: 2017.05.02
#

import numpy as np
import pandas as pd
from scipy.linalg.misc import norm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import sklearn.linear_model.logistic as logi

def load_data(file_path, **kw):
    ''' 
    =================== 读取数据 ==================
    输入:
        file_path   数据文件的路径
        **kw        一个包含读取方法的字典
            header      指定文件中作为列名的行号, None表示无列名
            sep         数据分隔符
            encoding    文件的编码方式
            index_col   指定文件中作为列指标的列号
    输出:
        data        读取到的数据作为二维矩阵返回
    ============================================
    '''
    data = pd.read_table(file_path, **kw)
    X = np.array(data.iloc[:, 0:-1].values[:,:])
    Y = np.array(data.iloc[:, -1].values[:])
    return X, Y


def Newton_iterate(X, Y, B, eps, max_it=10000):
    ''' 
    =================== 牛顿迭代 ====================
        输入:
            X       样例矩阵, 每行一个样例
            Y       样例类别, 向量
            B       初始参数值 (w; b)
            eps     容许的误差限
            max_it  最大迭代次数
        输出:
            B1      收敛的参数值 (w; b)
            i       实际迭代次数
    =============================================
    '''

    m = X.shape[0] # number of the samples
    X_hat = np.c_[X, np.ones(m)]

    def p(X, B):
        ''' p(y = 1 | X; B) '''
        tmp = np.exp(np.dot(B, X))
        return tmp / (1 + tmp)
    
    def plpB(X, Y, B):
        ''' l 对 B 的一阶偏导数 '''
        lst = [ X[i,:] * (p(X[i,:].T, B) - Y[i]) for i in range(m) ]
        return np.sum(lst, 0)
        
    def p2lpB2(X, Y, B):
        ''' l 对 B 的二阶偏导数 '''
        n = X.shape[1]
        _sum = np.zeros((n, n))
        for i in range(m):
            tmp = p(X[i,:], B)
            _sum += np.outer(X[i,:], X[i,:]) * tmp * (1 - tmp)
        return _sum
    
    B1 = B - np.dot(np.linalg.inv(p2lpB2(X_hat, Y, B)), plpB(X_hat, Y, B))

    i = 0
    while norm(B1 - B) > eps:
        B = B1
        B1 = B - np.dot(np.linalg.inv(p2lpB2(X_hat, Y, B)), plpB(X_hat, Y, B))
        i += 1
        if i > max_it:  # 迭代不收敛
            print("Not converge with ", max_it, "iterations.")
            print("Error norm: ", norm(B1 - B))
            print("(W, b): ", B1)
            exit()
        
    return B1, i


def sigmoid(w, x, b):
    ''' 对率函数 '''
    return 1.0 / (1 + np.exp(-np.dot(w, x) - b))


def show_wm(X, Y):
    ''' 绘制西瓜数据 '''
    Y = Y.astype(np.bool)
    plt.plot(X[Y,0], X[Y,1], "b*")
    plt.plot(X[~Y, 0], X[~Y, 1], "ro")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title("watermelon data")
    plt.savefig('wm_data.png')


def plot_wm(X, Y, B):
    ''' 绘制对数几率函数, 并显示西瓜点 '''
    x, y = np.ogrid[0:1:50j, 0:1:50j]
    z = 1 / (1 + np.exp(-(B[0] * x + B[1] * y + B[2])))
    
    ''' 绘制对率曲面图 '''
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap = plt.cm.Blues_r, alpha = 0.5)
    ax.set_xlabel('density')
    ax.set_ylabel('ratio_sugar')
    ax.set_zlabel('sign')

    ax.scatter(X[:,0], X[:,1], Y, s = 100, marker = "o")
    plt.savefig('logisReg.png')

    
def main():
    X, Y = load_data('../../wm_data.txt', encoding='gbk', index_col=0)
    B0 = np.array([0.5, 0.5, 0])    # B0 = (w; b)
    eps = 1e-6

#    show_wm(X, Y)    # 西瓜数据 3.0 可视化
    B1, i = Newton_iterate(X, Y, B0, eps)   # 牛顿迭代求解对数似然函数的极值

    print("Parameter: w = ", B1[0:2])
    print("Parameter: b = ", B1[2])
    print("Iteration steps: i = ", i)


#    plot_wm(X, Y, B1)   # 绘制对率函数的曲面并标记数据点

if __name__ == "__main__":
    main()
