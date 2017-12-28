from __future__ import print_function
import tensorflow as tf


def main():
    sess = tf.Session()
    # Setting weight and biases
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # loss
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    # training loop
    init = tf.global_variables_initializer()
    sess.run(init)  # reset values to incorrect defaults
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print('W: %s b: %s loss: %s'%(curr_W, curr_b, curr_loss))


if __name__ == '__main__':
    main()