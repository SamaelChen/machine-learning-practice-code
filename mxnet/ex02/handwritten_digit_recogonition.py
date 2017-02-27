import numpy as np
import os
import urllib
import gzip
import struct
import mxnet as mx
import matplotlib.pyplot as plt
from sklearn import datasets
import logging


def download_data(url, force_download=True):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    return fname


def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return (label, image)


path = 'http://yann.lecun.com/exdb/mnist/'
(train_lbl, train_img) = read_data(
    path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(
    path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
# plt.show()
print('label: %s' % (train_lbl[0:10],))


def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32) / 255


batch_size = 100
train_iter = mx.io.NDArrayIter(
    to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)
data = mx.sym.Variable('data')
data = mx.sym.Flatten(data=data)
fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type='relu')
fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type='relu')
fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
shape = {'data': (batch_size, 1, 28, 28)}
mx.viz.plot_network(symbol=mlp, shape=shape)
logging.getLogger().setLevel(logging.DEBUG)
model = mx.mod.Module(symbol=mlp, context=mx.gpu(), data_names=[
                      'data'], label_names=['softmax_label'])
model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
          optimizer_params={'learning_rate': 0.1}, eval_metric='acc', num_epoch=10)
plt.imshow(val_img[0], cmap='Greys_r')
prob = model.predict(val_img[0:2].astype(np.float32) / 255)[0]
assert max(prob) > 0.99, "Low prediction accuracy."
print('Classified as %d with probability %f' % (prob.argmax(), max(prob)))
valid_acc = model.score(val_iter, eval_metric='acc')
valid_acc
list(valid_acc)

model = mx.model.FeedForward(
    symbol=mlp, num_epoch=10, learning_rate=0.1, ctx=mx.gpu())
model.fit(X=train_iter, eval_data=val_iter,
          batch_end_callback=mx.callback.Speedometer(batch_size, 200))
plt.imshow(val_img[0], cmap='Greys_r')
prob = model.predict(val_img[0:1].astype(np.float32) / 255)[0]
assert max(prob) > 0.99, "Low prediction accuracy."
print('Classified as %d with probability %f' % (prob.argmax(), max(prob)))
