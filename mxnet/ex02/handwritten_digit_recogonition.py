import numpy as np
import os
import urllib
import gzip
import struct
import mxnet as mx
import matplotlib.pyplot as plt
from sklearn import datasets


data = datasets.load_digits()
plt.imshow(data.images[0], cmap='gray')
data
data.target[0]
imgs = data.images
val_lbl = data.target
tmp = imgs[0]
tmp.shape[0]
imgs.reshape(imgs.shape[0], 1, 8, 8).astype(np.float32) / 255
