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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


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
            tmp.load()
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

plt.imshow(imgs[1000], cmap='gray')
classes[1000]
len(imgs)
plt.imshow(imgs[10000], cmap='gray')
classes[10000]
for i in range(len(imgs)):
    imgs[i] = np.asarray(imgs[i].convert('L'))
imgs = np.array(imgs)
encoder = LabelEncoder()
# y = encoder.fit_transform(classes)
# Y = np_utils.to_categorical(y)
imgs[0]
imgs = np.array(imgs)
X = imgs.reshape(len(imgs), 32 * 32)
X = X / 255
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3)
y_train[0:4]
model = AdaBoostClassifier(DecisionTreeClassifier(
    max_depth=20, max_features=512, min_samples_leaf=50), n_estimators=150, learning_rate=0.05)
model.fit(X_train, y_train)

imags2 = []
path = '/home/samael/github/captcha-break/weibo.cn/spliter/dataset'
files = os.listdir(
    '/home/samael/github/captcha-break/weibo.cn/spliter/dataset')
files.sort()
for f in files:
    tmp = Image.open(path + '/' + f)
    tmp = np.asarray(tmp.convert('L'))
    imags2.append(tmp)

imags2[1]
imags2 = np.array(imags2)
X_val = imags2.reshape(len(imags2), 32 * 32)
X_val = X_val / 255
y_pred = model.predict(X_val)
y_pred
for i in range(len(y_pred)):
    plt.figure(figsize=(128, 128))
    plt.subplot(40, 1, i + 1)
    plt.imshow(imags2[i], cmap='gray')
    plt.title("Class {}".format(y_pred[i]))
