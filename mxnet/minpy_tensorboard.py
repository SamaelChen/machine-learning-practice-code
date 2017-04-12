from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import grad_and_loss
from examples.utils.data_utils import gaussian_cluster_generator as make_data
from minpy.context import set_context, gpu
import numpy

# from minpy.visualization.writer import LegacySummaryWriter as SummaryWriter
# import minpy.visualization.summaryOps as summaryOps
from tensorboard import FileWriter
from tensorboard.summary import scalar

summaries_dir = 'private/tmp/LR_log'

def predict(w, x):
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum
    return prob


def train_loss(w, x):
    prob = predict(w, x)
    loss = -np.sum(label * np.log(prob)) / num_samples
    return loss


"""Use Minpy's auto-grad to derive a gradient function off loss"""
grad_function = grad_and_loss(train_loss)
train_writer = FileWriter(summaries_dir + '/train')

# Using gradient descent to fit the correct classes.


def train(w, x, loops):
    for i in range(loops):
        dw, loss = grad_function(w, x)
        # gradient descent
        w -= 0.1 * dw
        if i % 10 == 0:
            print('Iter {}, training loss {}'.format(i, loss))
        # summary1 = scalar('loss', loss)
        # train_writer.add_summary(summary1, i)
        # print(loss)
        for ele in loss:
            summary1 = scalar('loss', ele)
            train_writer.add_summary(summary1, i)
    train_writer.flush()
    train_writer.close()


# Initialize training data.
num_samples = 10000
num_features = 500
num_classes = 5
data, label = make_data(num_samples, num_features, num_classes)
weight = random.randn(num_features, num_classes)
train(weight, data, 100)
