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
conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type='tanh')
pool1 = mx.sym.Pooling(data=tanh1, pool_type='max',
                       kernel=(2, 2), stride=(2, 2))

conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=20)
tanh2 = mx.sym.Activation(data=conv2, act_type='tanh')
pool2 = mx.sym.Pooling(data=tanh2, pool_type='max',
                       kernel=(2, 2), stride=(2, 2))

flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type='tanh')
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu(0))
lenet_model.fit(train_iter,  # train data
                eval_data=val_iter,  # validation data
                optimizer='sgd',  # use SGD to train
                # use fixed learning rate
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',  # report accuracy during training
                # output progress for each 100 data batches
                batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                num_epoch=100)  # train for at most 10 dataset passes
