################################################
#   coding: gbk
#
#   决策树 ( decision tree ): 
#  
#   《机器学习》 周志华 清华大学出版社 第4章 决策树 
#
#   Writen by Jarvis (zjw.math@qq.com)
#
#   Date: 2017.05.07
#

import numpy as np
import os
import csv
import json
from graphviz import Digraph
from os.path import join, dirname
from json import dumps
from sklearn import tree

class Bunch(dict):
    """ 用于保存数据的字典 """
    def __init__(self, **kwargs):

        return super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

def MapInit():
    """ 初始化西瓜数据映射表
    Parameters:
        无
    Returns:
        map: dict   西瓜的属性值与整数的映射字典
    """
    map = {}
    module_path = dirname(__file__)
    with open(join(module_path, 'wm_data', 'map.dat'), encoding='gbk') as f:
        for line in f:
            if line.startswith('*'):    # 属性以 * 开头
                lst = line.replace('\n', '').replace('*', '').split(' ')
                P_key = lst[0]
                map[P_key] = {  # 建立属性中英文以及编号的关联
                    'chs':lst[1], 
                    'num':lst[2]}
                continue
            value, key = tuple(line.replace('\n', '').split(' '))   # 属性值
            map[P_key][key] = value
    return map

def load_data(file, type=np.float):
    """ 载入数据
    Parameters:
        file: str           文件名
        type: type          数据类型
    Returns:
        Bunch对象:
            data: ndarray          数据值
            target: ndarray        分类值
            target_names: ndarray  类别
            feature_names: ndarray 属性
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'wm_data', file)) as csv_file:
        data_file = csv.reader(csv_file)

        # 首行记录了样本数, 属性数, 类别名称
        tmp = next(data_file)   
        nSamples = int(tmp[0])
        nFeatures = int(tmp[1])
        target_names = np.array(tmp[2:])

        # 第二行为属性名称
        tmp = next(data_file)   
        feature_names = np.array(tmp[1:-1])

        data = np.empty((nSamples, nFeatures), dtype=type)
        target = np.empty((nSamples, ), dtype=np.int)

        # 读取数据保存到 data 和 target 数组中
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[1:-1], dtype=type)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return Bunch(data=data, target=target, 
                target_names=target_names, 
                feature_names=feature_names)


class DecisionTree():
    """ 决策树类 
    包含了以下方法:
        load_data               导入数据
        ent                     计算信息熵
        gain                    计算信息增益
        gini                    计算基尼值
        gini_index              计算基尼指数
        fit                     训练数据
        image                   绘制决策树
        predict                 根据决策树预测
        postpruning             后剪枝
        train_C45_entropy       使用 sklearn 库函数以熵为标准分类
        train_C45_gini          使用 sklearn 库函数以基尼质数为标准分类
    """

    def __init__(self):
        return

    def load_data(self, file, type=np.float):
        """ 载入数据
        1. 初始化映射表 
        2. 从文件中读取数据
        3. 将属性名称和编号加入映射表
        """
        self.map = MapInit()
        data = load_data(file, type)
        self.data = data.data
        self.target = data.target
        self.target_names = data.target_names
        self.feature_names = data.feature_names
        self.k_feature = np.empty((self.data.shape[1],), dtype=np.int)
        for i in range(self.data.shape[1]):
            self.k_feature[i] = len(np.unique(self.data[:,i]))
        
        return

    def ent(self, y=None):
        """ 计算信息熵
                    |y|
        Ent(D) = - \SUM pk log2 pk
                    k=1
        Parameters:
            y: ndarray         类别数组, 默认为所有样本
        Returns:
            info_ent: np.float  信息熵
        """
        if y is None:
            y = self.target

        _, nDv = np.unique(y, return_counts=True)   # 统计各类别数量
        pk = nDv / y.shape[0]                       # 计算各类别占比
        info_ent = -np.sum(pk[i] * np.log2(pk[i]) for i in range(len(nDv)) if pk[i] != 0)

        return info_ent


    def gain(self, feature, idx=slice(None)):
        """ 计算信息增益
                                      n  |Dv|
        Gain(D, feature) = Ent(D) - \SUM ---- Ent(Dv)
                                     v=1 |D|
        Paramters:
            feature: int/str        样例属性
            idx: ndarray           样例的下标, 默认为所有样例
        Returns:
            info_gain: double       信息增益 (infomation gain)
        """
        if isinstance(feature, np.int32) or isinstance(feature, int):
            feature_idx = feature
        else:
            feature_idx = self.map[feature]
        X = self.data[idx, feature_idx]
        y = self.target[idx]
        
        nD = len(y)
        Dv, nDv = np.unique(X, return_counts=True)
        info_gain = self.ent(y) - np.sum(nDv[i] / nD * 
                                      self.ent(y[X == Dv[i]]) 
                                      for i in range(len(nDv)))
        return info_gain


    def gini(self, y=None):
        """ 计算基尼值
                   |y|  
        Gini(D) = \SUM \SUM  pk pk'
                   k=1 k'!=k
                       |y|
                = 1 - \SUM pk^2
                       k=1
        Parameters:
            y: ndarray         类别数组, 默认为所有样本
        Returns:
            gini: np.float      基尼值
        """
        if y is None:
            y = self.target

        _, nDv = np.unique(y, return_counts=True)   # 统计各类别数量
        pk = nDv / y.shape[0]                       # 计算各类别占比
        gini = 1 - np.sum(pk[i] ** 2 for i in range(len(nDv)))

        return gini


    def gini_index(self, feature, idx=slice(None)):
        """ 计算基尼指数
                                    n  |Dv|
        Gini_index(D, feature) =  \SUM ---- Gini(Dv)
                                   v=1 |D|
        Paramters:
            feature: int/str        样例属性
            idx: ndarray           样例的下标, 默认为所有样例
        Returns:
            gini_idx: double          基尼指数 (gini index)
        """
        if isinstance(feature, np.int32) or isinstance(feature, int):
            feature_idx = feature
        else:
            feature_idx = self.map[feature]
        X = self.data[idx, feature_idx]
        y = self.target[idx]

        nD = len(y)
        Dv, nDv = np.unique(X, return_counts=True)
        gini_idx = np.sum(nDv[i] / nD * self.gini(y[X == Dv[i]]) for i in range(len(nDv)))

        return gini_idx


    def __unique__(self, X):
        """ 判断二维数组 X 的所有行是否相同
        Parameters:
            X: ndarray     二维数组
        Returns:
            True/False
        """
        for i in range(X.shape[1]):
            if len(np.unique(X[:,i])) == 1:
                pass
            else:
                return False
        return True


    def __argmax__(self, array, random=False, artiSel=False, feature=None):
        """ 用于得到最大元素的指标
        Paramters:
            array: ndarray      待计算的数组
            random: bool        如果最大元素不止一个, 则返回时是否随机选择, 默认不随机
                                ::即返回找到的第一个指标
            artiSel: bool       是否人工选择最大值
            feature: ndarray    剩余的属性
        Returns:
            arg: int            最大元素的指标 
        """
        if not artiSel:
            if not random:
                return np.argmax(array)
            else:
                max_value = np.max(array)
                lst = [i for i in range(len(array)) if array[i] == max_value]
                return np.random.choice(lst)
        else:
            max_value = np.max(array)
            lst = [i for i in range(len(array)) if array[i] == max_value]
            idx = input("Please choice a index of the feature: " + 
                        str(self.feature_names[feature[lst]]) + 
                        " from 0 to " + 
                        str(len(lst) - 1) + " :\n")
            return lst[int(idx)]

    def __argmin__(self, array, random=False, artiSel=False, feature=None):
        """ 用于得到最小元素的指标
        Paramters:
            array: ndarray      待计算的数组
            random: bool        如果最小元素不止一个, 则返回时是否随机选择, 默认不随机
                                ::即返回找到的第一个指标
            artiSel: bool       是否人工选择最大值
            feature: ndarray    剩余的属性
        Returns:
            arg: int            最小元素的指标 
        """
        if not artiSel:
            if not random:
                return np.argmin(array)
            else:
                min_value = np.min(array)
                lst = [i for i in range(len(array)) if array[i] == min_value]
                return np.random.choice(lst)
        else:
            min_value = np.min(array)
            lst = [i for i in range(len(array)) if array[i] == min_value]
            idx = input("Please choice a index of the feature: " + 
                        str(self.feature_names[feature[lst]]) + 
                        " from 0 to " + 
                        str(len(lst) - 1) + " :\n")
            return lst[int(idx)]


    def fit(self, idx=None, feature=None, criterion="entropy", random=False, artiSel=False):
        """ 训练数据集
            仅使用信息增益作为集合划分的准则
            算法: 参考 P74 图 4.2
        Parameters:
            idx: ndarray        待训练数据的下标, 默认为所有样本
            feature: ndarray    用于分类的属性, 默认为所有属性
            criterion: str      分类标准, 默认为 "熵" , 可选 "gini"
            random: bool        分类指标相等时是否随机选择, 默认不随机, 选择第一个
            artiSel: bool       ( Artificial select ) 选择最优划分属性时, 如果分类指标相等, 是否人工选择, 默认由 random 决定
                                ::加入该选项是为了生成和书上一致的决策树, 便于继续学习
        Returns:
            tree: Bunch         决策树
        

        Example of tree:
            {"Texture": {'samples': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
                         1: {"Root": {'samples': [ 0,  1,  2,  3,  4,  5,  7,  9, 14]
                                      1:  "Good",
                                      2: {"Color": {'samples': [ 5,  7, 14]
                                                    1:  "Good",
                                                    2: {"Touch": {'samples': [ 7, 14]
                                                                  1: "Good",
                                                                  2: "Bad"
                                                                 }
                                                       },
                                                    3:  "Good"
                                                   }
                                         },
                                       3:  "Bad"
                                      }
                             },
                         2: {"Touch": {'samples': [ 6,  8, 12, 13, 16]
                                       1:  "Bad",
                                       2:  "Good"
                                      }
                            },
                         3:  "Bad"
                        }
            }
        """

        # 参数预处理
        if feature is None:
            feature = np.array(range(self.data.shape[1]), np.int)
        if idx is None:
            idx = np.array(range(self.data.shape[0]), np.int)

        X = self.data[idx,:][:,feature]
        y = self.target[idx]
        cla, num = np.unique(y, return_counts=True)

        # 建立一棵空树
        tree = Bunch()

        # 如果样本全属于同一类别
        # 将 node 标记为该类叶节点 返回
        if len(cla) == 1:
            return self.target_names[y[0]]

        # 如果 feature 为空 or 样本在属性集上的取值相同
        # 将 node 标记为叶节点, 类别标记为样本中样本数最多的类 返回
        if X.shape[1] == 0 or self.__unique__(X):
            return self.target_names[int(cla[num == np.max(num)][-1])]

        # 选择最优划分属性
        if criterion == "entropy":
            info_gain = np.empty((X.shape[1],), np.float)
            for i in range(X.shape[1]):
                info_gain[i] = self.gain(feature[i], idx)
            if not artiSel: # 非人工选择
                opt_feature = self.__argmax__(info_gain, random=random, artiSel=False)
            else:           # 人工选择
                opt_feature = self.__argmax__(info_gain, artiSel=True, feature=feature)
        elif criterion == "gini":
            gini_idx = np.empty((X.shape[1],), np.float)
            for i in range(X.shape[1]):
                gini_idx[i] = self.gini_index(feature[i], idx)
            if not artiSel: # 非人工选择
                opt_feature = self.__argmin__(gini_idx, random=random, artiSel=False)
            else:           # 人工选择
                opt_feature = self.__argmin__(gini_idx, artiSel=True, feature=feature)
        else:
            raise ValueError("Unknown value \"" + criterion + "\"", "in DecisionTree.py")

        opt_feature_in_all = feature[opt_feature]


        fname = self.feature_names[opt_feature_in_all]
        tree[fname] = {'samples': idx}

        # 用最优属性划分样本子集
        for i in range(self.k_feature[opt_feature_in_all]):
            lst_samples = idx[X[:,opt_feature] == i + 1]
            lst_feature = feature[feature != opt_feature_in_all]
            if lst_samples.shape[0] == 0:
                tree[fname][i+1] = self.target_names[int(cla[num == np.max(num)][-1])]
            else:
                tree[fname][i+1] = self.fit(lst_samples, lst_feature)

        return tree


    def __image_link__(self, dot, name, parent_name, parent, children):
        """ 递归函数: 
            用于连接 Graphviz 图的父亲和儿子结点
        Parameters:
            dot: Digraph        有向图对象
            name: int           结点名, 这里统一用数字表示(因为在图中显示的是结点的 label 而非结点名
                                    所以直接用数字表示方便命名, 也方便递归)
            parent_name: str    父结点名
            parent: str         父结点的 label
            children: dict      子结点集
        Returns:
            name: int           当前层的函数结束后下一个结点名
        """
        for key in children:                        # key 是连接父亲和儿子的边 label
            if key == 'samples':
                continue
            if isinstance(children[key], str):      # 叶子结点
                dot.node(str(name), self.map[children[key]]['chs'], fontname="SimSun")
                dot.edge(parent_name, str(name), self.map[parent][str(key)], fontname="STKaiti")
                name += 1
                continue
            for subkey in children[key]:            # 非叶子结点, 这里实际上只有一组键值
                dot.node(str(name), self.map[subkey]['chs'], shape='box', fontname="SimSun")
                dot.edge(parent_name, str(name), self.map[parent][str(key)], fontname="STKaiti")
                name = self.__image_link__(dot, name+1, str(name), subkey, children[key][subkey])
        
        return name


    def image(self, tree, title, format='png'):
        """ Graphviz 绘图函数
        Parameters:
            tree: Bunch     需要绘制的字典树
            title: str      树名, 同时作为输出的文件名
            format: str     输出文件的格式, 默认为 'png'
        Returns:
            绘图并输出到 'image\' 文件夹
                filename.gv         使用 dot 语言的 Graphviz 图形描述文件
                filename.gv.png     绘制出的箭头图
            注意: 可以在命令行下使用 dot -Tpng -O filename.gv 来解释图形描述文件并生成 png 图
                 可用的输出格式有 -Tpdf -Tpng 等
        """
        dot = Digraph(comment=title, format=format)
        name = 0
        if isinstance(tree, str):       # 只有一个根节点
            dot.node(str(name), self.map[tree]['chs'], fontname="SimSun")
        else:                           # 根节点非叶子
            for key in tree:
                dot.node(str(name), self.map[key]['chs'], shape='box', fontname="SimSun")
                self.__image_link__(dot, name+1, str(name), key, tree[key])

        dot.render("image\\" + title + ".gv", view=True)

        return


    def predict(self, tree, idx):
        """ 利用决策树预测
        Paramters:
            tree: Bunch         决策树
            idx: ndarray        测试集指标
        Returns:
            _Y: ndarray/int     预测结果
        """
        if isinstance(idx, np.ndarray):
            test_set = self.data[idx,:]
        elif isinstance(idx, int):
            test_set = self.data[[idx],:]
        else:
            raise ValueError("Test set should be one or two dimention array")

        res_set = []
        for x in test_set:
            ptr = tree
            while True:
                if isinstance(ptr, str):    # 到达了叶子结点
                    res_set.append(ptr)
                    break
                else:   # 依照属性进行判断
                    for key in ptr: # 仅有一个键值对: 一个划分属性
                        fvalue = x[int(self.map[key]['num'])]
                    ptr = ptr[key][fvalue]

        return res_set if len(res_set) > 1 else res_set[0]


    def __postpruning__(self, tree, parent, pkey, idx, root):
        """ 后剪枝递归函数
        Paramters:
            tree: Bunch         决策树
            parent: Bunch       父结点
            pkey: int           tree 在父结点下的 key
            idx: ndarray        测试集指标
        Returns:
            tree: Bunch         剪枝后的树
        """
        if isinstance(tree, str):   # 遍历尽头
            return tree             # tree是叶子结点(分类), 不能剪枝, 直接返回
        for key in tree:            # key是非叶子结点(属性)
            if True in [isinstance(tree[key][subkey], dict) for subkey in tree[key]]:   # key不是最低层属性, 需要继续深入
                for subkey in tree[key]:
                    if subkey == 'samples':
                        continue
                    tree[key][subkey] = self.__postpruning__(tree[key][subkey], tree[key], subkey, idx, root)
            else:   # key是最低层的属性, 尝试剪枝
                # 计算剪枝前的精度
                predict_value = np.array([True if cls == 'Good' else False for cls in self.predict(root, idx)])
                real_value = np.array(self.target[idx], dtype=np.bool)
                tmp = predict_value ^ real_value    # 相同为False, 不同为True
                pre_accur = 1.0 * tmp[tmp == False].shape[0] / tmp.shape[0]
                
                # 备份分支并剪枝
                bak_tree = tree.copy()
                cla, num = np.unique(self.target[tree[key]['samples']], return_counts=True)
                parent[pkey] = self.target_names[int(cla[num == np.max(num)][-1])]

                # 计算剪枝后的精度
                predict_value = np.array([True if cls == 'Good' else False for cls in self.predict(root, idx)])
                real_value = np.array(self.target[idx], dtype=np.bool)
                tmp = predict_value ^ real_value    # 相同为False, 不同为True
                post_accur = 1.0 * tmp[tmp == False].shape[0] / tmp.shape[0]

                if pre_accur > post_accur:  # 如果剪枝前精度高, 则撤销剪枝
                    parent[pkey] = bak_tree
                else:                       # 如果剪枝后精度高, 则确认剪枝
                    pass
        return parent[pkey]

    def postpruning(self, tree, idx):
        """ 后剪枝接口
        对树进行后序遍历
        Paramters:
            tree: Bunch         决策树
            idx: ndarray        测试集指标
        Returns:
            tree: Bunch         剪枝后的树
        """
        Dumyhead = {'dumykey': tree}
        
        # 计算剪枝前的精度
        predict_value = np.array([True if cls == 'Good' else False for cls in self.predict(tree, idx)])
        real_value = np.array(self.target[idx], dtype=np.bool)
        tmp = predict_value ^ real_value    # 相同为False, 不同为True
        pre_accur = 1.0 * tmp[tmp == False].shape[0] / tmp.shape[0]

        print("Accuracy before postpruning: %.2f%%" % (pre_accur * 100))
        new_tree = self.__postpruning__(tree, Dumyhead, 'dumykey', idx, tree)
        
        # 计算剪枝后精度
        predict_value = np.array([True if cls == 'Good' else False for cls in self.predict(tree, idx)])
        real_value = np.array(self.target[idx], dtype=np.bool)
        tmp = predict_value ^ real_value    # 相同为False, 不同为True
        post_accur = 1.0 * tmp[tmp == False].shape[0] / tmp.shape[0]

        print("Accuracy after postpruning:  %.2f%%" % (post_accur * 100))

        return new_tree

    def train_C45_entropy(self, title):
        """ sklearn 库中的 C4.5 算法 
            使用 "entropy" 作为集合划分的依据
        """
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(self.data, self.target)
        with open("image\\" + title + ".gv", "w") as f:
            f = tree.export_graphviz(clf, out_file=f, 
                                     feature_names=self.feature_names, 
                                     class_names=self.target_names, 
                                     filled=True, rounded=True,
                                     special_characters=True)
        os.system("dot -Tpng -O image\\" + title + ".gv")
        os.system("start image\\" + title + ".gv.png")

        return clf


    def train_C45_gini(self, title):
        """ sklearn 库中的 C4.5 算法 
            使用 "gini" 作为集合划分的依据
        """
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.data, self.target)
        with open("image\\" + title + ".gv", "w") as f:
            f = tree.export_graphviz(clf, out_file=f, 
                                     feature_names=self.feature_names, 
                                     class_names=self.target_names, 
                                     filled=True, rounded=True,
                                     special_characters=True)
        os.system("dot -Tpng -O image\\" + title + ".gv")
        os.system("start image\\" + title + ".gv.png")

        return clf

