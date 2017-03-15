import random
import sys 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_input

mnist = mnist_input.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 40], stddev=0.3))
b1 = tf.Variable(tf.truncated_normal([40], stddev=0.3))
h = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([40, 10], stddev=0.3))
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.3))
y = tf.matmul(h, W2) + b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

sess.run(tf.global_variables_initializer())

for epoch in range(3000):
    trainx, trainy = mnist.train.next_batch(100)
    sess.run(train, {x: trainx, y_: trainy})
    print(epoch, np.mean(np.argmax(trainy, 1) == sess.run(tf.argmax(y, 1), {x: trainx})))

result = sess.run(tf.argmax(y, 1), {x: mnist.validation.images})
print ' '.join(map(str, result))

