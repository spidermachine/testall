#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    # image_resized = tf.image.rgb_to_grayscale(image_resized)
    return image_resized, label

"""
def get_train_dataset(img_dir):
    def input_fc():

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
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels))).map(_parse_function)
        dataset.repeat().batch(100)
        return dataset.make_one_shot_iterator().get_next()

    return input_fc

"""
def get_train_dataset(img_dir):
    def input_fc():

        files = os.listdir(img_dir)
        images = []
        # labels = []
        labelmap = {}
        ll = 0
        # lll = 0
        labels = []
        for label in files:
            if os.path.isdir(img_dir + "/" + label):
                for jpgfile in os.listdir(img_dir + "/" + label):
                    if jpgfile.split(".")[1] != "jpg":
                        print(jpgfile)
                        exit(0)
                    images.append(os.path.join(img_dir + "/" + label + "/", jpgfile))
                    key = label
                    if labelmap.get(key) is None:
                        labelmap[key] = ll
                        ll = ll + 1
                    labels.append(labelmap.get(key))
        print(
            labelmap
        )
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels))).map(_parse_function)
        dataset = dataset.shuffle(10000).batch(100).repeat(100)
        return dataset.make_one_shot_iterator().get_next()

    return input_fc




def get_test_dataset(img_dir):
    def input_fc():

        files = os.listdir(img_dir)
        images = []
        labels = []
        labelmap = {}
        ll = 0
        # lll = 0
        labels = []
        for bmpfile in files:
            if bmpfile.split(".")[1] != "jpg":
                print(bmpfile)
                exit(0)
            images.append(os.path.join(img_dir + "/", bmpfile))
            key = bmpfile.split("_")[1][0]
            if labelmap.get(key) is None:
                labelmap[key] = ll
                ll = ll + 1
            labels.append(labelmap.get(key))
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels))).map(_parse_function)
        dataset = dataset.batch(100)
        return dataset.make_one_shot_iterator().get_next()

    return input_fc



def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    # tf.logging.info(features)
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # tf.logging.info(input_layer)
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    tf.logging.info(dense)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=33)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    tf.logging.info(labels)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=32)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# def export_func():
#     inputs = tf.placeholder(dtype=tf.string, name="input_image")
#     feature_config = {'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string)}
#     tf_example = tf.parse_example(inputs, feature_config)
#     patch_images = tf.map_fn(tf.image.decode_image, tf_example["image/encoded"], dtype=tf.float32)
#     patch_images = tf.reshape(patch_images, [-1, 28, 28, 1])
#     receive_tensors = tf.placeholder(dtype=tf.float32, shape=[])
#     features = {"input": patch_images}
    # return tf.estimator.export.ServingInputReceiver(patch_images, inputs)
dataset_path = './imgdownload'

def main(unused_argv):
    # Load training and eval data

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    #
    mnist_classifier.train(
        input_fn=get_train_dataset(dataset_path),
        steps=100000,
        hooks=[logging_hook])

    # mnist_classifier.export_savedmodel('export', serving_input_receiver_fn=export_func)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=get_train_dataset(dataset_path))
    print(eval_results)

import time
if __name__ == "__main__":
    start = time.mktime(time.localtime())
    tf.app.run()
    print(time.mktime(time.localtime()) - start)
    # main(None)
    # get_train_dataset('/Users/zkp/Desktop/for_zkp/train')()
