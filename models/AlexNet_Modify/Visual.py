#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization of the training accurancy.

@author: Jarvis ZHANG
@date: 2017/8/1
@framework: Tensorflow
@editor: VS Code
"""

import re
import os
import sys
import numpy as np
# Install SimSun font first, then modify the matplotlib fonts
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "SimSun"

step = []
accu = []

with open(os.path.join(sys.path[0], "output"), 'r') as reader:
    lines = reader.readlines()
    final_line = lines[-1]
    lines = lines[:-1]
    pattern1 = re.compile("\d{5}")
    pattern2 = re.compile("\d\.\d{4}")
    pattern3 = re.compile("\d\.\d{3}")
    for line in lines:
        step.append(pattern1.search(line).group())
        accu.append(pattern2.search(line).group())
    test_accu = pattern3.search(final_line).group()

step = np.array(step)
accu = np.array(accu)
plt.plot(step, accu)
plt.plot([step[0]], [test_accu], color='red')
plt.title("训练过程中损失的变化")
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.xlim([0, 3000])
plt.legend(["Training loss", "Test accurancy: " + test_accu])
plt.show()
