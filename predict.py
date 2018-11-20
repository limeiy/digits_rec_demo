# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as plt # plt用于显示图片
import matplotlib.image as mpimg # mpimg用于读取图片
import numpy as np
import os
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import PIL
import tensorflow as tf
# import config
import traceback
import requests
import json


import sys
import threading

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 28

def process_pic(file_name, mnist_pic):
    img = Image.open(file_name)
    img = img.convert('L')
    #img.show()

    if img.size[0] != IMAGE_SIZE or img.size[1] != IMAGE_SIZE:
        print("image size %d*%d resize to 28*28" % img.size)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE),Image.ANTIALIAS)

    #img.show()

    #  setup a converting table with constant threshold
    # threshold  =   200
    # table  =  []
    # for  i  in  range( 256 ):
    #   if  i  <  threshold:
    #      table.append(0)
    #   else :
    #      table.append( 1 )
    #  convert to binary image by the table
    #img  =  img.point(table,  '1')

    # img = img.convert('1')

    data = []
    for i in range(IMAGE_SIZE):
        print_value = ''
        for j in range(IMAGE_SIZE):
            #pixel = img.getpixel((j,i))
            #print_value += str(pixel) + " "
            # mnist 里的颜色是0代表白色（背景），1.0代表黑色
            if mnist_pic == True:
                new_pixel = float(img.getpixel((j, i))) / 255.0
            else:
                new_pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
            #new_pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
            data.append(new_pixel)
        #print(print_value+"\n")
    arr = np.array(data, dtype=float)
    #print(arr)
    #arr = np.reshape(np.array(data),[-1,IMAGE_SIZE*IMAGE_SIZE])
    return arr
    #
    #
    # img.show()
    # img.save('new_img.png')
    #
    # lena = mpimg.imread('new_img.png')
    # print(lena.shape)
    # plt.imshow(lena,cmap='Greys')
    # plt.axis('off')
    # plt.show()


def predict(data, model_path, model_name):
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.path.join(model_path, model_name))
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    # Access saved Variables directly
    #print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict ={x:data}

    #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("y:0")

    result=sess.run(op_to_restore,feed_dict)
    print(result)
    print(sess.run(tf.argmax(result,1)))
    #This will print 60 which is calculated

def predict_deep(data, model_path, model_name):
    sess=tf.Session()
    #First let's load meta graph and restore weights

    saver = tf.train.import_meta_graph(os.path.join(model_path, model_name))
    saver.restore(sess,tf.train.latest_checkpoint(model_path))

    # Access saved Variables directly
    #print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
    feed_dict ={x:data, keep_prob:1.0}

    #Now, access the op that you want to run.
    y_conv = graph.get_tensor_by_name("fc2/y_conv:0")

    result=sess.run(y_conv,feed_dict)
    print(result)
    label = sess.run(tf.argmax(result,1))
    print(label)
    return label[0]
    #This will print 60 which is calculated

def predict_by_train_set():
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
    train_images = mnist.train.images

    import matplotlib.pyplot as plt  # plt用于显示图片
    plt.imshow(train_images[1].reshape(28,28),cmap='Greys')
    plt.show()
    #predict([train_images[1]])
    predict_deep([train_images[1]])

def predict_by_mnist_pics(file_name):
    data = process_pic(file_name, mnist_pic=True)

    import matplotlib.pyplot as plt  # plt用于显示图片
    plt.imshow(data.reshape(28, 28), cmap='Greys')
    plt.show()
    predict_deep([data], config.MODEL_SAVE_PATH, config.MODEL_SAVE_NAME+'.meta')

def predict_by_self_pics(file_name):
    data = process_pic(file_name, mnist_pic=False)

    # import matplotlib.pyplot as plt  # plt用于显示图片
    # plt.imshow(data.reshape(28, 28), cmap='Greys')
    # plt.show()
    result = predict_deep([data],config.MODEL_SAVE_PATH, config.MODEL_SAVE_NAME+'.meta')
    return result

##############################################
# 以下为采用tensorflow serving 方式
##############################################
class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()
    self._predict_result = {}

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      #print(self._done)
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

  def record_predict_result(self, no, prediction):
    self._predict_result[no]=prediction

  def get_predict_results(self):
    # print(self._done)
    # print(self._num_tests)
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._predict_result


def _create_rpc_callback(no, label, result_counter):
  """Creates RPC callback function.
  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      print("prediction is: ", prediction)
      print("label is: ", label)
      if label != None and label != prediction:
        result_counter.inc_error()
      result_counter.record_predict_result(no, prediction)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference_grpc(hostport, work_dir, concurrency, images, labels):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  num_tests = images.shape[0]

  host, port = hostport.split(':')
  print(host)
  print(port)
  try:
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
  except e:
    print(e)
    traceback.print_exc()
  for i in range(num_tests):
    image = images[i]
    label = None
    if labels is not None:
      print("labels[i]",labels[i])
      label = numpy.argmax(labels[i])
      print("label",label)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist_deep_demo'      # !!注意这里的名字要跟serving的MODEL_NAME一一对应
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, image.size]))
    request.inputs['keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))  # 不能直接设置
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(i, label, result_counter))

  if labels is None:
    return result_counter.get_predict_results()

  return result_counter.get_error_rate()


#restapi mode
def do_inference(hostport, work_dir, concurrency, images, labels):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  num_tests = images.shape[0]
  result_counter = _ResultCounter(num_tests, concurrency)

  for i in range(num_tests):
    image = images[i]
    label = None
    if labels is not None:
      print("labels[i]",labels[i])
      label = numpy.argmax(labels[i])
      print("label",label)

    input_data={}
    input_data['images']=image.tolist()
    input_data['keep_prob']=1.0

    result_counter.throttle()

    try:
      json_data = {"signature_name": 'predict_images', "instances": [input_data]}
      print(json_data)
      result = requests.post(hostport, json=json_data)
      print(result)
      print(result.text)
    except:
      traceback.print_exc()
      result_counter.inc_error()
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      predict_result = json.loads(result.text)
      print(predict_result)
      prediction = numpy.argmax(predict_result['predictions'][0])
      print("prediction is: ", prediction)
      print("label is: ", label)
      if label != None and label != prediction:
        result_counter.inc_error()
      result_counter.record_predict_result(i, prediction)
      result_counter.inc_done()
      result_counter.dec_active()

  if labels is None:
    return result_counter.get_predict_results()

  return result_counter.get_error_rate()

