##################################################################
#   coding: gbk
#
#   10折交叉验证法 ( Cross Validation ) , 留一法 ( Left-One-Out ): 
#
#   《机器学习》 周志华 清华大学出版社
#
#   第3章 线性模型 习题3.4
#
#   目标: 比较10折交叉验证法和留一法所估计出的对率回归的错误率
#
#   Writen by Jarvis (zjw.math@qq.com)
#
#   Date: 2017.05.03
#

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


def show_class(X, Y, tX, tY, logreg, attr, attr1, attr2):
    
    Pt_colors = ('R', 'Y', 'B')
    Bg_colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(Bg_colors[:len(np.unique(Y))])
    h = .02  # 网格步长

    # 绘制边界
    # 网格点 [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果绘制为彩图
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(17, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

    # 绘制训练的数据点
    s = []
    for i in range(3):
        ss = plt.scatter(X[Y == i, 0], X[Y == i, 1], c=Pt_colors[i], s=100, marker=(i+3, 0))
        s.append(ss)
        ss = plt.scatter(tX[tY == i, 0], tX[tY == i, 1], s=100, facecolors='W', marker=(i+3, 0))
        s.append(ss)

    plt.xlabel(attr[attr1])
    plt.ylabel(attr[attr2])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend((s[0], s[2], s[4], s[1], s[3], s[5]), 
               ("Iris Setosa (training)", "Iris Versicolour (training)", "Iris Virginica (training)", 
                "Iris Setosa (testing)", "Iris Versicolour (testing)", "Iris Virginica (testing)"), 
               scatterpoints=1,
               loc="upper left",
               fontsize=12)

#    plt.show()
    plt.savefig("iris" + str(attr1) + str(attr2) + ".png")
    plt.close()


def main(f, attr1=0, attr2=1):
    iris = datasets.load_iris()         # 读取iris数据
    _X = iris.data[:,[attr1, attr2]]            # 选择两个属性, 便于绘图
    _Y = iris.target
    attr = iris.feature_names
    kf = KFold(n_splits=10, shuffle=True, random_state=10)   # 10折划分, 打乱

    # 10折交叉验证
    i, flag = 0, True
    ErrorRate_CV = np.ones((10,))
    for train, test in kf.split(_X):
        X, Y = _X[train, :], _Y[train]
        tX, tY = _X[test, :], _Y[test]

        logreg = LogisticRegression(C=1e5)

        # 创建一个分类器的实例并拟合数据
        logreg.fit(X, Y)

        # 用测试集测试数据
        preY = logreg.predict(np.c_[tX[:,0], tX[:,1]])
        ErrorRate_CV[i] = preY[preY != tY].shape[0] / tY.shape[0]
        i += 1

        if flag:
            show_class(X, Y, tX, tY, logreg, attr, attr1, attr2)
            flag = False

    # 留一法验证
    ErrorRate_LOO = np.empty((_Y.shape[0],), dtype=np.int)
    for i in range(_Y.shape[0]):
        train = np.array([j for j in range(_Y.shape[0]) if j != i])
        X, Y = _X[train, :], _Y[train]
        tX, tY = _X[i, :], _Y[i]

        logreg = LogisticRegression(C=1e5)

        # 创建一个分类器的实例并拟合数据
        logreg.fit(X, Y)

        # 用测试集测试数据
        preY = logreg.predict(np.c_[tX[0], tX[1]])
        ErrorRate_LOO[i] = 0 if preY == tY else 1
        i += 1

    f.write(attr[attr1] + ' - ' + attr[attr2] + '\n')
    f.write("10 Fold Cross Validation (error rate): " + str(np.mean(ErrorRate_CV)) + '\n')
    f.write("Left One Out (error rate)            : " + str(np.mean(ErrorRate_LOO)) + '\n\n')


if __name__ == '__main__':
    with open("result.txt", "w") as f:
        main(f, 0, 1)
        main(f, 1, 2)
        main(f, 2, 3)
