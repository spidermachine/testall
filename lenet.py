import tensorflow as tf
import os
import numpy as np


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_bmp(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    image_resized = tf.image.per_image_standardization(image_resized)
    return image_resized, label


def get_train_dataset(img_dir):
    files = os.listdir(img_dir)
    images = []
    labelmap = {}
    ll = 0
    labels = []
    for bmpfile in files:
        if bmpfile.split(".")[1] != "bmp":
            print(bmpfile)
            exit(0)
        images.append(os.path.join(img_dir + "/", bmpfile))
        key = bmpfile.split("_")[1][0]
        if labelmap.get(key) is None:
            labelmap[key] = ll
            ll = ll + 1
        labels.append(labelmap.get(key))
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels))).map(_parse_function)
    dataset.repeat().shuffle(100).batch(2)
    return dataset.make_one_shot_iterator().get_next()

image, label = get_train_dataset('template_7')
with tf.Session() as sess:
    try:
        while True:
            # images = sess.run(image)
            # tf.reshape(images, [-1, 28,28,1])
            # print(image.shape)
            labels = sess.run(label)
            print(labels)
            # break
    except tf.errors.OutOfRangeError:
        print("end!")
