import random
import sys 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_input

mnist = mnist_input.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

ximg = tf.reshape(x, [-1, 28, 28, 1])[:, 4:24, 4:24, :]

ximg = tf.nn.max_pool(ximg, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 8], stddev=0.3))
b1 = tf.Variable(tf.constant(0.3 * np.ones([5, 5, 8]), dtype=tf.float32))

h_i = tf.nn.conv2d(ximg, W1, [1, 1, 1, 1], 'SAME')

h_i = tf.nn.max_pool(h_i, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME') + b1

h = tf.nn.relu(tf.reshape(h_i, [-1, 5*5*8]))

W2 = tf.Variable(tf.truncated_normal([5*5*8, 10], stddev=0.3))
b2 = tf.Variable(tf.constant(0.3 * np.ones([10]), dtype=tf.float32))
y = tf.matmul(h, W2) + b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

sess.run(tf.global_variables_initializer())

for epoch in range(6000):
    trainx, trainy = mnist.train.next_batch(100)
    sess.run(train, {x: trainx, y_: trainy})
    if epoch % 100 == 0:
        print(epoch, np.mean(np.argmax(trainy, 1) == sess.run(tf.argmax(y, 1), {x: trainx})))

result = sess.run(tf.argmax(y, 1), {x: mnist.validation.images})
incorrect = result != np.argmax(mnist.validation.labels, 1)
print np.mean(1 - incorrect)

wrong_disp = np.where(incorrect)[0][:20]
f, axarr = plt.subplots(4, 5)

for ind, i in enumerate(wrong_disp):
    axarr[ind/5, ind%5].imshow(mnist.validation.images[i].reshape([28, 28]))
    axarr[ind/5, ind%5].set_title('Predicted {0}'.format(str(result[i])))
plt.show()
