#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf
import cv2

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层的variables和ops
W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
L1_relu = tf.nn.relu(L1_conv + b_conv1)
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义第二个卷积层的variables和ops
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
L2_relu = tf.nn.relu(L2_conv + b_conv2)
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(L2_pool, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout层
W_fc2 = tf.Variable(tf.truncated_normal([1024, 32], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[32]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义优化器和训练op
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def _parse_function(filename, label):
    print(filename.eval())
    image_string = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image_string = cv2.resize(image_string, (28,28), interpolation=cv2.INTER_CUBIC)
    image_string = image_string / 255
    image_string = np.reshape(image_string, [-1, 784])
    print(image_string)
    return image_string, label
    # return image_resized, tf.one_hot(label, 32, dtype=tf.int32)


def get_train_dataset(img_dir):
    files = os.listdir(img_dir)
    images = []
    # labels = []
    labelmap = {}
    ll = 0
    lll = 0
    labels = np.array([[0] * 32 for i in range(len(files))])
    for bmpfile in files:
        images.append(os.path.join(img_dir + "/", bmpfile))
        key = bmpfile.split("_")[1][0]
        if labelmap.get(key) is None:
            labelmap[key] = ll
            ll = ll + 1
        labels[lll][labelmap[key]] = 1
        lll = lll + 1
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels)))
    dataset.repeat().batch(100)
    return dataset.make_one_shot_iterator().get_next()

images, labels = get_train_dataset('template_7')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 执行训练迭代
    for it in range(1000):
        while True:
            try:
                train_step.run(
                    feed_dict={x: images, y_: labels, keep_prob: 0.5})
            except Exception as e:
                break

    print('完成训练!')
