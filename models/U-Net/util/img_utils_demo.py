#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
from img_utils import *


def show_example():
    img = plt.imread(os.path.join(sys.path[0], "double.jpg"))
    plt.imshow(img)
    plt.show()

def affine_demo():
    p = np.array([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = np.array([
        [42, 211], [125, 155], [255, 100],
        [80, 235], [190, 235], [85, 295], [180, 295]
    ])
    image = plt.imread(os.path.join(sys.path[0], "mr_big_ori.jpg"))
    plt.subplot(231)
    plt.imshow(image)
    plt.title("原图")
    transformed_image = mls_affine_deformation(image, p, q, alpha=1, density=1)
    plt.subplot(232)
    plt.imshow(transformed_image)
    plt.title("仿射变换 -- 采样密度 1")
    transformed_image = mls_affine_deformation_inv(image, p, q, alpha=1, density=1)
    plt.subplot(233)
    plt.imshow(transformed_image)
    plt.title("逆仿射变换 -- 采样密度 1")
    transformed_image = mls_affine_deformation(image, p, q, alpha=1, density=0.7)
    plt.subplot(235)
    plt.imshow(transformed_image)
    plt.title("仿射变换 -- 采样密度 0.7")
    transformed_image = mls_affine_deformation_inv(image, p, q, alpha=1, density=0.7)
    plt.subplot(236)
    plt.imshow(transformed_image)
    plt.title("逆仿射变换 -- 采样密度 0.7")
    plt.xlim((0, image.shape[1]))
    plt.ylim((image.shape[0], 0))

    plt.show()

def affine_demo2():
    p = np.array([
        [80, 92], [97, 92], [81, 96], [89, 96], [99, 96]
    ])
    q = np.array([
        [80, 92], [97, 92], [81, 96], [89, 96], [99, 96]
    ])
    image = plt.imread(os.path.join(sys.path[0], "monalisa_ori.jpg"))
    plt.subplot(121)
    plt.imshow(image)
    transformed_image = mls_affine_deformation_inv(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.imshow(transformed_image)
    plt.show()


affine_demo2()