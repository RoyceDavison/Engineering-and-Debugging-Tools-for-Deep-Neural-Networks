from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import models.research.slim.nets.resnet_utils as resnet_utils
from cleverhans.dataset import MNIST
from cleverhans.model_zoo.resNet import ResNet
from cleverhans.utils import AccuracyReport

resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim

"""No-op context manager."""

class NoOpScope(object):
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

optimizer = tf.train.AdamOptimizer (learning_rate=1e-4)

####Fetch Data#####
tf.reset_default_graph()

tf.set_random_seed(1234)
sess = tf.Session()

train_start = 0
train_end = 1000
test_start = 1001
test_end = 1200

mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')
img_rows, img_cols, nchannels = x_train.shape[1:4]

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 2))

nb_classes = y_train.shape[1] #? Y_train.shape[1]
nb_filters = 64 #?

model1 = ResNet(scope = "model1", nb_classes = nb_classes, nb_filters = nb_filters)

"""
logits, _ = model1.fprop(x).O_LOGITS

loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
train_op = optimizer.minimize(loss)

dataset_size=x_train.shape[0]
indices = list(range(dataset_size))
random_seed = 100
# print(indices)
np.random.seed(random_seed)
np.random.shuffle(indices)

init_op = tf.global_variables_initializer()
sess.run(init_op)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

###########Train the model##########
#epoch = 100
epoch = 1
batch_size = 64
iter_num = dataset_size // batch_size
# ind

select_index = np.array(indices)
odd_Array = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# even_Array = np.array([1,0,1,0,1,0,1,0,1,0])
odd_Matrix = np.repeat(odd_Array[np.newaxis, :], batch_size, axis=0)
# print(odd_Matrix)

print (x_train.shape)
for n in range (epoch):
    print (n)
    indices = list (range (dataset_size))
    random_seed = n
    np.random.seed (random_seed)
    np.random.shuffle (indices)

    for i in range (iter_num):
        select = select_index[i * batch_size:i * batch_size + batch_size]
        x_batch = x_train[select]
        y_batch = y_train[select]
        #         print(y_batch)
        y_mul = odd_Matrix * y_batch
        y_one = np.sum (y_mul, axis=1).astype (int)
        y_two = np.zeros ((batch_size, 2))

        y_two[np.arange (batch_size), y_one] = 1

        #         print(y_one)
        #         print(y_two)

        #         print(y_two.shape)
        #         print(x_batch.shape)
        #         print(y_one.shape)
        sess.run (train_op, feed_dict={x: x_batch, y: y_two})

saver = tf.train.Saver (max_to_keep=10, keep_checkpoint_every_n_hours=0.5)
saver.save (sess, 'res_50_model.ckpt')
print("Finish the training.")

    #####################################
    #####################################
    #####################################
    #####################################
    #####################################

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

eval_params = {'batch_size': batch_size}
"""