import numpy as np
import tensorflow as tf
from utils import *
import numpy as np

sess = tf.InteractiveSession()

h, w, c = 256, 256, 1
n_classes = 4

# Parameters
learning_rate = 0.0001
training_iters = 1000
batch_size = 100
step_display = 10
step_save = 500
path_save = 'convnet'

# Network Parameters
dropout = 0.5 # Dropout, probability to keep units

keep_prob = tf.placeholder(tf.float32)

# Construct dataloader
loader = DensityLoader()

x = tf.placeholder(tf.float32, [None, h, w, c])
# x_flat = tf.reshape(x, [-1, h*w*c])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, n_classes])
b_fc2 = bias_variable([n_classes])

y_conv= tf.matmul(h_fc1_drop, W_fc2) + b_fc2

init = tf.initialize_all_variables()

if True:
    # Initialization
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())
    minibatch_size = 512

    for i in range(200000):
        images_batch, labels_batch = loader.next_batch(minibatch_size)
        if i%10 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x:images_batch, y_: labels_batch, keep_prob: 1.0})
          print("step %d, training accuracy %g"%(i, train_accuracy))

          train_loss = cross_entropy.eval(feed_dict={
              x:images_batch, y_: labels_batch, keep_prob: 1.0})
          print("step %d, training loss %g"%(i, train_loss))

        train_step.run(feed_dict={x: images_batch, y_: labels_batch, keep_prob: 0.5})

        if i%30 == 1:
            test_acc_sum = 0.0
            test_steps = len(images_test)/batch_size
            for test_step in range(test_steps):
                test_acc_sum += sess.run(accuracy, feed_dict={x: images_test[test_steps*batch_size:(test_steps+1)*batch_size],
                                              y: labels_test[test_steps*batch_size:(test_steps+1)*batch_size],
                                              keep_prob: 1.})
            print("Test Accuracy: {}".format(test_acc_sum/test_steps))

