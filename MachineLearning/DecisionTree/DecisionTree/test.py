################################################
#   coding: gbk
#
#   决策树 ( decision tree ): 
#  
#   《机器学习》 周志华 清华大学出版社 第4章 决策树 
#
#   1. 导入西瓜数据 2.0
#   2. 基于信息增益和基尼指数训练决策树
#   3. 根据得到的决策树对新的样例可以进行预测
#   4. 对得到的决策树进行剪枝处理
#
#   Writen by Jarvis (zjw.math@qq.com)
#
#   Date: 2017.05.07
#

from DecisionTree import DecisionTree
import numpy as np


#dt = DecisionTree()
#dt.load_data('wm2.0.csv', np.int)

# 4.2 划分选择
# 4.2.1 信息增益
#dec_tree = dt.fit(criterion="entropy")
#dt.image(dec_tree, "info_gain")

# 4.2.2 增益率

# 4.2.3 基尼指数
#dec_tree2 = dt.fit(criterion="gini")
#dt.image(dec_tree2, "gini_index")

"""
可以看出, 在这个例子中基尼指数和信息增益的划分结果是相同的
"""

# 4.3 剪枝处理
# 4.3.1 预剪枝 --> 太烦懒得写了

# 4.3.2 后剪枝
#training_data = np.array([0,1,2,5,6,9,13,14,15,16])
#test_data = np.array([3,4,7,8,10,11,12])
#dec_tree3 = dt.fit(training_data, artiSel=True)
#dt.image(dec_tree3, "Nonpruning_tree")
#dec_tree3_pruning = dt.postpruning(dec_tree3, test_data)
#dt.image(dec_tree3_pruning, "Pruning_tree")


# 使用 sklearn 库函数生成决策树
#dt2 = DecisionTree()
#dt2.load_data('wm3.0a.csv')

#clf = dt2.train_C45_entropy("entropy")

dt3 = DecisionTree()
dt3.load_data('wm3.0a.csv')

clf = dt3.train_C45_gini("gini")
