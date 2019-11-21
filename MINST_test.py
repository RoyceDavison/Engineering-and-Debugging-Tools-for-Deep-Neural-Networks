from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils

from IPython.display import HTML
import torchvision.datasets as dset

import tensorflow as tf

import models
import models.research
import models.research.slim
import models.research.slim.nets as nets

from torch.utils.data.sampler import SubsetRandomSampler

import models.research.slim.nets.resnet_utils as resnet_utils


resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.
  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.
  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.
  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.
  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode. If this is set
      to None, the callers can specify slim.batch_norm's is_training parameter
      from an outer slim.arg_scope.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
      activations at the last unit of each block and store them in the
      `outputs_collections` before subsampling them. This gives us access to
      higher resolution intermediate activations which are useful in some
      dense prediction problems but increases 4x the computation and memory cost
      at the last unit of each block.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes a non-zero integer, net contains the
      pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with (slim.arg_scope([slim.batch_norm], is_training=is_training)
            if is_training is not None else NoOpScope()):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                                              store_non_strided_activations)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points
resnet_v1.default_image_size = 224


def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
  blocks = [
      resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                      stride=2),
      resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                      stride=2),
      resnet_v1_block('block3', base_depth=depth_func(256), num_units=6,
                      stride=2),
      resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                      stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_50.default_image_size = resnet_v1.default_image_size


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  store_non_strided_activations=False,
                  min_base_depth=8,
                  depth_multiplier=1,
                  reuse=None,
                  scope='resnet_v1_101'):
  """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
  depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
  blocks = [
      resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                      stride=2),
      resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                      stride=2),
      resnet_v1_block('block3', base_depth=depth_func(256), num_units=23,
                      stride=2),
      resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                      stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_101.default_image_size = resnet_v1.default_image_size


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  min_base_depth=8,
                  depth_multiplier=1,
                  reuse=None,
                  scope='resnet_v1_152'):
  """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
  depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
  blocks = [
      resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                      stride=2),
      resnet_v1_block('block2', base_depth=depth_func(128), num_units=8,
                      stride=2),
      resnet_v1_block('block3', base_depth=depth_func(256), num_units=36,
                      stride=2),
      resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                      stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)

resnet_v1_152.default_image_size = resnet_v1.default_image_size


def resnet_v1_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  min_base_depth=8,
                  depth_multiplier=1,
                  reuse=None,
                  scope='resnet_v1_200'):
  """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
  depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
  blocks = [
      resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                      stride=2),
      resnet_v1_block('block2', base_depth=depth_func(128), num_units=24,
                      stride=2),
      resnet_v1_block('block3', base_depth=depth_func(256), num_units=36,
                      stride=2),
      resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                      stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)

resnet_v1_200.default_image_size = resnet_v1.default_image_size
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

################
###Fetch data###
################
from cleverhans.dataset import MNIST

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

#logits, _ = resnet_v1_50(inputs=x,num_classes=2, reuse=True)
logits, _ = resnet_v1_50(inputs=x,num_classes=2)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
train_op = optimizer.minimize(loss)

print(x_train.shape)
dataset_size=x_train.shape[0]
indices = list(range(dataset_size))
random_seed = 100
np.random.seed(random_seed)
np.random.shuffle(indices)

init_op = tf.global_variables_initializer()
sess.run(init_op)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


#epoch = 100
epoch = 1
batch_size = 64
iter_num = dataset_size # batch_size
# ind

select_index = np.array (indices)
odd_Array = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# even_Array = np.array([1,0,1,0,1,0,1,0,1,0])
odd_Matrix = np.repeat(odd_Array[np.newaxis, :], batch_size, axis=0)
# print(odd_Matrix)

print (x_train.shape)
print("--------Training start-------")
for n in range(epoch):
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
        sess.run(train_op, feed_dict={x: x_batch, y: y_two})

saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=0.5)
saver.save(sess, 'res_50_model.ckpt')

#####################
##Attack with CLW2###
#####################
from cleverhans.attacks.carlini_wagner_l2 import CarliniWagnerL2
from cleverhans.model import Model as model
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy
from cleverhans.train import train

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()
eval_params = {'batch_size': batch_size}
BACKPROP_THROUGH_ATTACK = False

def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr (report, report_key, acc)
    if is_adv is None:
        report_text = None
    elif is_adv:
        report_text = 'adversarial'
    else:
        report_text = 'legitimate'
    if report_text:
        print ('Test accuracy on %s examples: %0.4f' % (report_text, acc))


# Create a new model and train it to be robust to FastGradientMethod
#model2 = ModelBasicCNN('model2', nb_classes, nb_filters)

##我们遇到的问题
#model2 = resNet()
#cw2 = CarliniWagnerL2 (model2, sess=sess)
#adv_x = cw.generate(x, **cw_params)
#adv_x = cw2.generate (x)
#preds_adv = model.get_logits (adv_x)

def attack(x):
    #return cw2.generate(x, **fgsm_params)
    return cw2.generate(x)

  loss2 = CrossEntropy(model2, smoothing=label_smoothing, attack=attack)
  #preds2 = model2.get_logits(x)
  adv_x2 = attack(x)

  if not BACKPROP_THROUGH_ATTACK:
    # For the fgsm attack used in this tutorial, the attack has zero
    # gradient so enabling this flag does not change the gradient.
    # For some other attacks, enabling this flag increases the cost of
    # training, but gives the defender the ability to anticipate how
    # the atacker will change their strategy in response to updates to
    # the defender's parameters.
    adv_x2 = tf.stop_gradient(adv_x2)
  preds2_adv = model2.get_logits(adv_x2)

  def evaluate2():
    # Accuracy of adversarially trained model on legitimate test inputs
    do_eval(preds2, x_test, y_test, 'adv_train_clean_eval', False)
    # Accuracy of the adversarially trained model on adversarial examples
    do_eval(preds2_adv, x_test, y_test, 'adv_train_adv_eval', True)

  # Perform and evaluate adversarial training
  train(sess, loss2, x_train, y_train, evaluate=evaluate2,
        args=train_params, rng=rng, var_list=model2.get_params())

  # Calculate training errors
  if testing:
    do_eval(preds2, x_train, y_train, 'train_adv_train_clean_eval')
    do_eval(preds2_adv, x_train, y_train, 'train_adv_train_adv_eval')

  return report