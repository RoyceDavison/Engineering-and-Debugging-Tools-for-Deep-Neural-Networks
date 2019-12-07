from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import models.research.slim.nets.resnet_utils as resnet_utils
from cleverhans.dataset import MNIST
from cleverhans.model_zoo.resNet import ResNet
from cleverhans.utils import AccuracyReport
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import CarliniWagnerL2

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
y = tf.placeholder(tf.float32, shape=(None, 10))

nb_classes = 10 #? Y_train.shape[1]
nb_filters = 64 #?

model = ResNet(scope = "model1", nb_classes = nb_classes, nb_filters = nb_filters)
preds = model.get_logits(x)
loss = CrossEntropy(model, smoothing=0.1)
print("Defined TensorFlow model graph.")

loss = CrossEntropy(model, smoothing=0.1)
print("Defined TensorFlow model graph.")

###########################################################################
# Training the model using TensorFlow
###########################################################################

#training ResNetmodel



# from cleverhans.compat import flags
# FLAGS = flags.FLAGS

VIZ_ENABLED = True
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100
import os
MODEL_PATH = os.path.join('models', 'mnist')
TARGETED = True
source_samples = SOURCE_SAMPLES
attack_iterations = ATTACK_ITERATIONS

# Train an MNIST model
train_params = {
  'nb_epochs': 1,
  'batch_size': 128,
  'learning_rate': 0.001,
  'filename': os.path.split(MODEL_PATH)[-1]
}

from cleverhans.train import train
from cleverhans.utils_tf import model_eval
rng = np.random.RandomState([2017, 8, 30])
# check if we've trained before, and if we have, use that pre-trained model
train(sess, loss, x_train, y_train, args=train_params, rng=rng)

cw = CarliniWagnerL2(model, sess=sess)

saver = tf.train.Saver()
saver.save(sess, MODEL_PATH)

adv_inputs = np.array(
          [[instance] * nb_classes for
           instance in x_test[:source_samples]], dtype=np.float32)
one_hot = np.zeros((nb_classes, nb_classes))
one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1
adv_inputs = adv_inputs.reshape(
    (source_samples * nb_classes, img_rows, img_cols, nchannels))
adv_ys = np.array([one_hot] * source_samples,
                  dtype=np.float32).reshape((source_samples *
                                             nb_classes, nb_classes))
yname = "y_target"
cw_params_batch_size = source_samples * nb_classes
cw_params = {'binary_search_steps': 1,
             yname: adv_ys,
             'max_iterations': attack_iterations,
             'learning_rate': CW_LEARNING_RATE,
             'batch_size': cw_params_batch_size,
             'initial_const': 10}

adv = cw.generate_np(adv_inputs,
                     **cw_params)

eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
adv_accuracy = model_eval(
        sess, x, y, preds, adv, adv_ys, args=eval_params)
print(adv_accuracy)
print('--------------------------------------')

#
# # Evaluate the accuracy of the MNIST model on legitimate test examples
# eval_params = {'batch_size': batch_size}
# accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
# assert x_test.shape[0] == test_end - test_start, x_test.shape
# print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
# report.clean_train_clean_eval = accuracy

###########################################################################
# Craft adversarial examples using Carlini and Wagner's approach
###########################################################################
# nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
# print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
#     ' adversarial examples')
# print("This could take some time ...")
#
# # Instantiate a CW attack object
# cw = CarliniWagnerL2(model, sess=sess)

"""
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
"""