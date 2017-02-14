import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def read_img(path):
    dirs = os.listdir(path)
    imgs = []
    classes = []
    for d in dirs:
        tmp_path = path + '/' + d
        files = os.listdir(tmp_path)
        for f in files:
            tmp = Image.open(tmp_path + '/' + f)
            tmp = np.asarray(tmp.convert('L'))
            imgs.append(tmp)
            classes.append(d)
    return(imgs, classes)


imgs, classes = read_img(
    '/home/samael/github/hexo-practice-code/tensorflow/ex01/training_set')
imgs = np.array(imgs)
plt.imshow(imgs[1000], cmap='gray')
classes[1000]
encoder = LabelEncoder()
y = encoder.fit_transform(classes)
Y = np_utils.to_categorical(y)
imgs[0]
X = imgs.reshape(len(imgs), 32 * 32)
X = X / 255
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Sequential()
model.add(Dense(512, input_shape=(1024,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(14))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10,
          verbose=1, validation_data=(X_test, Y_test))

imags2 = []
path = '/home/samael/github/captcha-break/weibo.cn/spliter/dataset'
files = os.listdir(
    '/home/samael/github/captcha-break/weibo.cn/spliter/dataset')
for f in files:
    tmp = Image.open(path + '/' + f)
    tmp = np.asarray(tmp.convert('L'))
    imags2.append(tmp)

imags2[1]
imags2 = np.array(imags2)
X_val = imags2.reshape(len(imags2), 32 * 32)
X_val = X_val / 255
Y_pred = model.predict(X_val)
Y_pred.shape
index = []
for item in Y_pred:
    # print(np.argmax(item))
    index.append(np.argmax(item))

Y_res = encoder.inverse_transform(index)
for i in range(len(Y_res)):
    plt.figure(figsize=(128, 128))
    plt.subplot(40, 1, i + 1)
    plt.imshow(imags2[i], cmap='gray')
    plt.title("Class {}".format(Y_res[i]))
plt.imshow(imgs[1], cmap='gray')
imgs[1].shape
tmp = Image.open(path + '/' + files[0])
tmp.rotate(30)
tmp2 = tmp.rotate(30)
tmp2
tmp2 = np.asarray(tmp2.convert('L'))
plt.imshow(tmp2, cmap='gray')
tmp2[0]
tmp1 = np.asarray(tmp.convert('L'))


def get_board(img):
    x = []
    y = []
    tmp = []
    for i in range(32):
        if np.any(img[i] != np.repeat(255, 32)):
            tmp.append(i)
    x.append(np.min(tmp) - 1)
    x.append(np.max(tmp) + 1)
    tmp = []
    for i in range(32):
        if np.any(img[:, i] != np.repeat(255, 32)):
            tmp.append(i)
    y.append(np.min(tmp) - 1)
    y.append(np.max(tmp) + 1)
    return(x, y)


x, y = get_board(tmp1)
tmp3 = tmp1.copy()
tmp3[x[0], y[0]] = 144
tmp3[x[0], y[1]] = 144
tmp3[x[1], y[0]] = 144
tmp3[x[1], y[1]] = 144
plt.imshow(tmp3, cmap='gray')
plt.imshow(tmp1, cmap='gray')
