#####################################################
#   coding: gbk
#
#   线性判别分析 ( Linear Discriminant Analysis )
#
#   机器学习 周志华 清华大学出版社
#
#   第3章 线性模型 习题3.5
#
#   目标: 实现线性判别分析, 并给出西瓜数据集 3.0 alpha 上的结果
#
#   Writen by Jarvis (zjw.math@qq.com)
#
#   Date: 2017.05.03
#

import numpy as np
import matplotlib.pylab as plt


def load_data(file_path, usecols, delim='\t', dtype='float', skiprows=0):
    ''' 
    =================== 读取数据 ==================
    输入:
        file_path   数据文件的路径
        usecols     读取的列号
        delim       数据分隔符, 默认为 '\t'
        dtype       数据类型, 默认为 'float'
        skiprows    开头跳过的行数, 默认不跳行
    输出:
        data        读取到的数据作为二维矩阵返回
    ============================================
    '''
    data = np.loadtxt(file_path, delimiter=delim, usecols=usecols, dtype=dtype, skiprows=skiprows)
    return data

def LDA(X1, X2):
    '''
    ================== 线性判别分析 ==================
    输入:
        X1  样例数据集 正例
        X2  样例数据集 反例
    输出:
        w   投影直线的系数
    =================================================
    '''
    
    # np.cov把每行当作一个变量
    Sw = np.cov(X1.T) * (X1.shape[0] - 1) + np.cov(X2.T) * (X2.shape[0] - 1)
    # 计算判别系数
    w = np.dot(np.linalg.inv(Sw), np.mean(X2, axis=0) - np.mean(X1, axis=0))
    return w


def plot_wm(X1, X2, w):
    ''' 绘制西瓜数据散点图以及投影直线 '''
    plt.scatter(X1[:,0], X1[:,1], s=100, marker=(4, 0), facecolors="B")
    plt.scatter(X2[:,0], X2[:,1], s=100, marker=(3, 0), facecolors="R")
    x = np.linspace(0, 1, 20)
    y = -w[0] * x / w[1]    # w[0]*x + w[1]*y = 0  <==  W^T.X = 0
    plt.plot(x, y)

    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title('LDA')
    plt.savefig("LDA.png")
    plt.show()


def main():
    data = load_data('../../wm_data.txt', np.arange(1, 4), skiprows=1)
    X, Y = data[:, 0:2], data[:, 2]
    Y = Y.astype(np.bool)
    X1, X2 = X[Y, :], X[~Y, :]
    w = LDA(X1, X2)

    print("Coefficient: ", w)
    plot_wm(X1, X2, w)


if __name__ == "__main__":
    main()