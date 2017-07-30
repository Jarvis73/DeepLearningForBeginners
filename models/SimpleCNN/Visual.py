#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization of the training accurancy.

@author: Jarvis ZHANG
@date: 2017/7/30
@framework: Tensorflow
@editor: VS Code
"""

import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "SimSun"

step = []
accu = []

with open(os.path.join(sys.path[0], "output"), 'r') as reader:
    lines = reader.readlines()
    final_line = lines[-1]
    lines = lines[:-1]
    pattern1 = re.compile("\d{5}")
    pattern2 = re.compile("0\.\d{4}")
    for line in lines:
        step.append(pattern1.search(line).group())
        accu.append(pattern2.search(line).group())
    test_accu = pattern2.search(final_line).group()

step = np.array(step)
accu = np.array(accu)
plt.plot(step, accu)
plt.plot([step[0], step[-1]], [test_accu] * 2, color='red')
plt.title("训练过程中验证精度的变化")
plt.xlabel("迭代次数")
plt.ylabel("验证精度")
plt.xlim([0, 20000])
plt.ylim([0.95, 1.0])
plt.legend(["Validation accurancy", "Test accurancy"])
plt.show()
