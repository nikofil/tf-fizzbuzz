import tensorflow as tf
import numpy as np

numenc = lambda x: [] if x == 0 else numenc(x / 2) + [x % 2]

digits = 10

def num_encode(i):
    x = numenc(i)
    return [0]*(digits - len(x)) + x

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

train_set = [(num_encode(x), fizzbuzz(x)) for x in range(100,1000)]
test_set = [(num_encode(x), fizzbuzz(x)) for x in range(100,1000)]

x = tf.placeholder(shape=[None, digits], dtype=tf.float32)
w1 = tf.Variable(tf.random_normal(shape=[digits, 100]))
h = tf.nn.relu(tf.matmul(x, w1))
w2 = tf.Variable(tf.random_normal(shape=[100, 4]))
y = tf.matmul(h, w2)
y_ = tf.placeholder(shape=[None, 4], dtype=tf.float32)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_task = tf.train.GradientDescentOptimizer(0.08).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

for _ in range(10000):
    perm = np.random.permutation(train_set)
    train_x = map(lambda a: np.array(a, dtype=np.float32), perm[:,0])
    train_y = map(lambda a: np.array(a, dtype=np.float32), perm[:,1])
    for i in range(0, len(train_x), batch_size):
        sess.run(train_task, {x: train_x[i:i+batch_size], y_: train_y[i:i+batch_size]})
    print(_, np.mean(sess.run(tf.argmax(y, 1), {x: train_x}) == np.argmax(train_y, 1)))
