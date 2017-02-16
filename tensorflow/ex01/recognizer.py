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
import copy


def read_img(path):
    dirs = os.listdir(path)
    imgs = []
    classes = []
    for d in dirs:
        tmp_path = path + '/' + d
        files = os.listdir(tmp_path)
        for f in files:
            tmp = Image.open(tmp_path + '/' + f)
            # tmp = np.asarray(tmp.convert('L'))
            imgs.append(tmp)
            classes.append(d)
    return(imgs, classes)


def rotate_img(img, theta):
    tmp = copy.deepcopy(img)
    for i in range(32):
        for j in range(32):
            if tmp.getpixel((i, j)) == 0:
                tmp.putpixel((i, j), 144)
    tmp = tmp.rotate(theta)
    for i in range(32):
        for j in range(32):
            if tmp.getpixel((i, j)) == 0:
                tmp.putpixel((i, j), 255)
    for i in range(32):
        for j in range(32):
            if tmp.getpixel((i, j)) == 144:
                tmp.putpixel((i, j), 0)
    return(tmp)


imgs, classes = read_img(
    '/home/samael/github/hexo-practice-code/tensorflow/ex01/training_set')
for i in range(len(imgs)):
    for delta_theta in range(-10, 11):
        imgs.append(rotate_img(imgs[i], theta=delta_theta))
        classes.append(classes[i])

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
