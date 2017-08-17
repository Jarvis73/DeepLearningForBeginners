#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet demo

@author: Jarvis ZHANG
@date: 2017/8/14
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
from outer_utils import *
from unet import Unet, Trainer
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['image.cmap'] = 'gist_earth'
log_dir = "log_dir"
log_path = os.path.join(sys.path[0], log_dir)

#show_one_example()

nx, ny = 572, 572
generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)
channels = generator.channels
n_class = generator.n_class

net = Unet(channels=channels, n_class=n_class, features_root=16, layers=3)

trainer = Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2, learning_rate=0.2))

path = trainer.train(generator, training_iters=20, epochs=10, display_step=2, restore=False, write_graph=False)

x_test, y_test = generator(1)

prediction = net.predict(os.path.join(log_path, "model.cpkt"), x_test)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.9
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()
fig.savefig(os.path.join(sys.path[0], "toy_problem.png"))
