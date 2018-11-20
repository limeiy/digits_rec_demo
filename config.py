MNIST_DATASET_PATH = './mnist_dataset'
MNIST_LOG_PATH = './log'
SUMMARY_LOG_DIR = 'mnist_deep_summaries'
MODEL_EXPORT_DIR = 'exported_model'

MODEL_SAVE_PATH = './models'
MODEL_SAVE_NAME = 'model.ckpt'

MNIST_DATASET_PICS_TRAIN_PATH = './mnist_dataset_pics/train'
MNIST_DATASET_PICS_TEST_PATH = './mnist_dataset_pics/test'
HANDWRITE_DIGITS_PICS_PATH = './handwrite_digits_pics'

#server = '172.18.144.99:9000'
#server = '36.111.85.107:8500'
server = 'http://number-testbdt.swgz.tae.ctyun.cn/v1/models/mnist_deep_demo:predict'
#server = 'http://36.111.85.107:8501/v1/models/mnist_deep_demo:predict' #restful api方式访问
work_dir = '/tmp'
concurrency = 1