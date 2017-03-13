import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2
import random
from io import BytesIO
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt


class OCRBatch(object):

    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


def gen_rand():
    buf = ""
    for i in range(4):
        buf += str(random.randint(0, 9))
    return buf


def get_label(buf):
    a = [int(x) for x in buf]
    return np.array(a)


def gen_sample(captcha, width, height):
    num = gen_rand()
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1 / 255.0)
    img = img.transpose(2, 0, 1)
    return (num, img)


class OCRIter(mx.io.DataIter):

    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha()

        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]

    def __iter__(self):
        for k in range(int(self.count / self.batch_size)):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height)
                data.append(img)
                label.append(get_label(num))

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']

            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


tmp = OCRIter(10, 8, 4, 30, 80)
tmp.captcha
for i in tmp:
    a = i
a
a.data
b = a.data
c = a.label

plt.imshow(b[0][1][2].asnumpy())
plt.imshow(b[0][1][0].asnumpy())
c[0].asnumpy()
