# coding: gbk

import kNN
import numpy as np


################################################################################
#                                                                              #
#                                    约会网站                                   #
#                                                                              #
################################################################################
#group, labels = kNN.createDataSet()
#tlabel = kNN.classify0([0, 0], group, labels, 3)
#print(tlabel)

# 导入约会对象的数据
datingDataMat, datingLabels = kNN.file2matrix('./data/datingTestSet2.txt')

# 用 Matplotlib 创建散点图
import matplotlib
import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], 
#           datingDataMat[:,2], 
#           15.0 * np.array(datingLabels), 
#           15.0 * np.array(datingLabels)
#           )
#plt.xlabel('玩游戏视频所耗时间百分比', fontproperties='SimHei')
#plt.ylabel('每周消费的冰淇淋公升数', fontproperties='SimHei')
#plt.show()

# 获取分类
all_cls = np.unique(datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)

# 绘图颜色
color = ['Blue', 'Yellow', 'Red']

# 按不同分类绘图
sca = [None] * len(all_cls)
for i, cls in enumerate(all_cls):
    sca[i] = ax.scatter(datingDataMat[datingLabels == cls, 0], 
                        datingDataMat[datingLabels == cls,1], 
                        15.0 * np.array(datingLabels[datingLabels == cls]), 
                        color[i % 3])

plt.xlabel('每年获取的飞行常客里程数')
plt.ylabel('玩游戏视频所耗时间百分比')
plt.legend(tuple(sca), ('不喜欢', '魅力一般', '极具魅力'))
plt.savefig('image/fig1.png')

# 归一化
#normMat, ranges, minVals = autoNorm(datingDataMat)

# 分类器测试
#kNN.datingClassTest()

# 约会网站预测
#kNN.classifyPerson()


