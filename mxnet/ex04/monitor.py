import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

# some constants
batch_size = 32
numb_bin_digits = 10
numb_softmax_outputs = 2**numb_bin_digits
numb_Training_samples = numb_softmax_outputs * 8
numb_Validation_samples = numb_softmax_outputs * 2
num_epoch = 10


def getBinaryArrayAndLabels(numberrows, numbBinDigits):
    '''
    Generate numberrows that will repeat all possible values of numbBinDigits
    (for instance 4 digits generate 16 vals ->0000 to 1111)
    :param numberrows: how many rows of data
    :param numbBinDigits: how many binary digits
    :return: a train array shape(numberrows, numbBinDigits) , and a label array(shape(numberrows,)
    '''
    largestBinNumb = 2**numbBinDigits
    binStringStartsHere = 2
    label = [row % largestBinNumb for row in range(numberrows)]
    train = [[int(bin(row % largestBinNumb)[binStringStartsHere:].zfill(numbBinDigits)[
                  col]) for col in range(0, numbBinDigits)] for row in range(0, numberrows)]
    return np.asarray(train), np.asarray(label)


# create our model

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(
    data=data, name='fc1', num_hidden=numb_softmax_outputs * 3)
act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(
    data=act1, name='fc2', num_hidden=numb_softmax_outputs * 2)
act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(
    data=act2, name='fc3', num_hidden=numb_softmax_outputs)
mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

# get train iterators
train_val, train_lab = getBinaryArrayAndLabels(
    numberrows=numb_Training_samples, numbBinDigits=numb_bin_digits)
itr_train = mx.io.NDArrayIter(
    train_val, train_lab, batch_size=batch_size, shuffle='True')

# get eval iterators
eval_val, eval_lab = getBinaryArrayAndLabels(
    numberrows=numb_Validation_samples, numbBinDigits=numb_bin_digits)
itr_val = mx.io.NDArrayIter(eval_val, eval_lab, batch_size=batch_size)

model = mx.mod.Module(context=mx.cpu(), symbol=mlp)


def norm_stat(d):
    return mx.nd.norm(d) / np.sqrt(d.size)


def early_stop(param):
    name_value = param.eval_metric.get_name_value()
    for name, value in name_value:
        if value > 0.001:
            return False


mon = mx.mon.Monitor(100, norm_stat)
model.fit(train_data=itr_train, eval_data=itr_val, monitor=mon,
          batch_end_callback=mx.callback.Speedometer(100, 100), num_epoch=10)
