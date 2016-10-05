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

# Construct dataloader
loader = DensityLoader()

x = tf.placeholder(tf.float32, [None, h, w, c])
x_flat = tf.reshape(x, [-1, h*w*c])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

W_fc1 = weight_variable([h*w*c, 512])
b_fc1 = bias_variable([512])
h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([512, n_classes])
b_fc2 = bias_variable([n_classes])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

keep_prob = tf.placeholder(tf.float32)

init = tf.initialize_all_variables()

if True:
    # Initialization
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())
    minibatch_size = 512
    print('start training')
    for i in range(20000):
        images_batch, labels_batch = loader.next_batch(minibatch_size)
        if i%2 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x:images_batch, y_: labels_batch, keep_prob: 1.0})
          print("step %d, training accuracy %g"%(i, train_accuracy))

          train_loss = cross_entropy.eval(feed_dict={
              x:images_batch, y_: labels_batch, keep_prob: 1.0})
          print("step %d, training loss %g"%(i, cross_entropy))

        train_step.run(feed_dict={x: images_batch, y_: labels_batch, keep_prob: 0.5})

        if i%2 == 0:
            images_test, labels_test = loader.load_test()
            print "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: images_test[:1000],
                                              y_:labels_test[:1000],
                                              keep_prob: 1.})

