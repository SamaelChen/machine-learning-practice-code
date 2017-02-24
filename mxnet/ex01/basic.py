import mxnet as mx
import minpy.numpy as np
import numpy
from minpy.core import grad
from minpy.core import grad_and_loss
from data_iter import SyntheticData
import logging


a = mx.nd.array([1, 2, 3])
b = mx.nd.array([[1, 2, 3], [4, 5, 6]])
c = np.array([1, 2, 3])
d = np.array([[1, 2, 3], [4, 5, 6]])
c
a.context
a
a.size
a.dtype
mx.nd.array(numpy.array([1, 2, 3]))
type(numpy.array([1, 2, 3]))
type(c)
np.ones((2, 3))
np.ones([2, 3])
mx.nd.ones([2, 3])
mx.nd.ones([2, 3]).asnumpy()


def foo(x):
    return(5 * (x**2) + 3 * x + 2)


print(foo(4))
d_foo = grad(foo)
d_l_foo = grad_and_loss(foo)
d_foo(4)
d_l_foo(4)

# Symbol
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b
# elemental wise times
d = a * b
# matrix multiplication
e = mx.sym.dot(a, b)
f = mx.sym.Reshape(d + e, shape=(1, 4))
# broadcast
g = mx.sym.broadcast_to(f, shape=(2, 4))
mx.viz.plot_network(symbol=g)

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=net, name='relu1', act_type='relu')
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(data=net, name='out')
mx.viz.plot_network(net, shape={'data': (100, 200)})


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                                 stride=stride, pad=pad, name='conv_%s%s' % (name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' % (name, suffix))
    act = mx.symbol.Activation(
        data=bn, act_type='relu', name='relu_%s%s' % (name, suffix))
    return act


prev = mx.symbol.Variable(name="Previos Output")
conv_comp = ConvFactory(data=prev, num_filter=64, kernel=(7, 7), stride=(2, 2))
shape = {"Previos Output": (128, 3, 28, 28)}
mx.viz.plot_network(symbol=conv_comp, shape=shape)

ex = c.bind(ctx=mx.cpu(), args={
            'a': mx.nd.ones([3, 3]), 'b': mx.nd.ones([3, 3]) * 2})
ex.forward()
print('number of outputs = %d\nthe first output = \n%s' %
      (len(ex.outputs), ex.outputs[0].asnumpy()))
ex.forward(is_train=True)
ex.outputs[0].asnumpy()

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = mx.sym.Variable('c')
d = a + b * c

data = mx.nd.ones((2, 3))
ex = d.bind(ctx=mx.cpu(), args={'a': data,
                                'b': data * 2, 'c': mx.nd.ones((2, 3)) * 3})
ex.forward()
ex.outputs[0].asnumpy()

# Train networks
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type='relu')
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(net, name='softmax')
data = SyntheticData(10, 128)
mx.viz.plot_network(net)
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])
logging.basicConfig(level=logging.INFO)
batch_size = 32
mod.fit(data.get_iter(batch_size), eval_data=data.get_iter(batch_size), optimizer='sgd',
        optimizer_params={'learning_rate': 0.1}, eval_metric='acc', num_epoch=5)
y = mod.predict(data.get_iter(batch_size))
y.shape
y
y.asnumpy()
