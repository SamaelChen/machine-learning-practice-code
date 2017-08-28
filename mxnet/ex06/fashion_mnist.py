import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import numpy as np
import gzip
import struct


def read_data(label, image):
    """
    Read data into numpy
    """
    base_url = '/home/samael/github/fashion-mnist/data/fashion/'
    with gzip.open(base_url + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(base_url + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    Reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32) / 255


def get_mnist_iter():
    """
    Create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, 100, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, 100)
    return (train, val)


batch_size = 100
train_iter, val_iter = get_mnist_iter()
data = mx.sym.var('data')
data = mx.sym.flatten(data=data)
# The first fully-connected layer and the corresponding activation function
fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
# MNIST has 10 classes
fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
# Softmax with cross entropy loss
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
mlp_model = mx.mod.Module(symbol=mlp, context=mx.gpu(0))
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              # use fixed learning rate
              optimizer_params={'learning_rate': 0.1},
              eval_metric='acc',  # report accuracy during training
              # output progress for each 100 data batches
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=100)  # train for at most 10 dataset passes
