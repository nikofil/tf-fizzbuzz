import tensorflow as tf
import numpy as np

numenc = lambda x: [] if x == 0 else numenc(x / 10) + [x % 10]
numdec = lambda x: reduce(lambda prev, cur: prev*10 + cur, x)

def fizzbuzz(x):
    if x % 3 == 0 and x % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif x % 3 == 0:
        return np.array([1, 0, 0, 0])
    elif x % 5 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([0, 0, 0, 1])
    
fblist = np.array(['fizz', 'buzz', 'fizzbuzz'])
fbdec = lambda x, y: fblist[list(x).index(1)] if x[-1] != 1 else y

train_set = [(numenc(x), fizzbuzz(x)) for x in range(100,1000)]
test_set = [(numenc(x), fizzbuzz(x)) for x in range(100,1000)]

x = tf.placeholder(shape=[None, 6], dtype=tf.float32)
w1 = tf.Variable(tf.random_normal(shape=[6, 10]))
b1 = tf.Variable(tf.random_normal(shape=[1, 10]))
h = tf.nn.relu(tf.matmul(x, w1) + b1)
w2 = tf.Variable(tf.random_normal(shape=[10, 4]))
b2 = tf.Variable(tf.random_normal(shape=[1, 4]))
y = tf.matmul(h, w2) + b2
y_ = tf.placeholder(shape=[None, 4], dtype=tf.float32)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_task = tf.train.GradientDescentOptimizer(0.08).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(10000):
    perm = np.random.permutation(train_set)
    train_x = map(lambda a: np.array([0] * (6-len(a)) + a, dtype=np.float32), perm[:,0])
    train_y = map(lambda a: np.array(a, dtype=np.float32), perm[:,1])
    sess.run(train_task, {x: train_x[:300], y_: train_y[:300]})
