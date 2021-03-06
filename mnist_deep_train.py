# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import argparse
import sys
import tempfile
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

tf.app.flags.DEFINE_integer('training_iteration', 20000,
                            'Number of training iterations.')
tf.app.flags.DEFINE_string('data_dir', './mnist_dataset',
                            'Directory for storing input data')
tf.app.flags.DEFINE_string('work_dir', '/tmp/mnist_deep_demo',
                            'Working directory.')
tf.app.flags.DEFINE_string('summary_log_dir', os.path.join(tf.app.flags.FLAGS.work_dir, 'summaries'),
                            'Summaries log directory')
tf.app.flags.DEFINE_string('model_export_dir', os.path.join(tf.app.flags.FLAGS.work_dir, 'exported_model'),
                            'Directory for storing exported model')
tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')
tf.app.flags.DEFINE_string('model_save_dir', os.path.join(tf.app.flags.FLAGS.work_dir, 'saved_model'),
                            'Directory for storing saved model')
tf.app.flags.DEFINE_string('model_save_name', 'model.ckpt',
                            'Name for the saved model')

FLAGS = tf.app.flags.FLAGS


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
      W_conv1 = weight_variable([5, 5, 1, 32])
      variable_summaries(W_conv1)
    with tf.name_scope('bias'):
      b_conv1 = bias_variable([32])
      variable_summaries(b_conv1)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    tf.summary.histogram('h_conv1',h_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram('h_pool1', h_pool1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    with tf.name_scope('weights'):
      W_conv2 = weight_variable([5, 5, 32, 64])
      variable_summaries(W_conv2)
    with tf.name_scope('bias'):
      b_conv2 = bias_variable([64])
      variable_summaries(b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    tf.summary.histogram('h_conv2', h_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram('h_pool2', h_pool2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    with tf.name_scope('weights'):
      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      variable_summaries(W_fc1)
    with tf.name_scope('bias'):
      b_fc1 = bias_variable([1024])
      variable_summaries(b_fc1)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('h_fc1', h_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.histogram('h_fc1_drop', h_fc1_drop)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    with tf.name_scope('weights'):
      W_fc2 = weight_variable([1024, 10])
      variable_summaries(W_fc2)
    with tf.name_scope('bias'):
      b_fc2 = bias_variable([10])
      variable_summaries(b_fc2)

    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2, name='y_conv')
    tf.summary.histogram('y_conv', y_conv)

  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def main(_):
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name="x")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # graph_location = tempfile.mkdtemp()
  # print('Saving graph to: %s' % graph_location)
  # train_writer = tf.summary.FileWriter(graph_location)
  # train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:

    print('Saving summaries to: %s' % FLAGS.summary_log_dir)
    train_writer = tf.summary.FileWriter(FLAGS.summary_log_dir + '/train', sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for i in range(FLAGS.training_iteration):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        summary,train_accuracy = sess.run([merged,accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # train_accuracy = accuracy.eval(feed_dict={
        #     x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

        train_writer.add_summary(summary, i)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('Done training!')
    train_writer.close()

    model_saved_path = os.path.join(FLAGS.model_save_dir, FLAGS.model_save_name)
    print('Save trained model to: ', model_saved_path)
    saver = tf.train.Saver()
    saver.save(sess, model_saved_path)
    print('Done Saving!')

    # Export model
    export_path_base = FLAGS.model_export_dir
    export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to: ', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y_conv)

    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x, 'keep_prob':tensor_info_keep_prob},
        outputs={'scores': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    #legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        'predict_images':
          prediction_signature
      }
    )
    builder.save()     # !!! MUST HAVE

    print('Done exporting!')

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    # 采用小一点的batch，避免out of memory错误
    for i in range(10):
      testSet = mnist.test.next_batch(50)
      print("test accuracy %g" % accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0}))



if __name__ == '__main__':
  tf.app.run()