from __future__ import print_function
import tensorflow as tf


def main():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    sess = tf.Session()
    print(sess.run([node1, node2]))

    # Multiple node conjunction
    node3 = tf.add(node1, node2)
    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)
    print(sess.run(adder_node, {a: 12, b: 22}))
    print(sess.run(adder_node, {a: [1, 2], b: [3, 1]}))
    add_and_triple = adder_node * 3.
    print(sess.run(add_and_triple, {a: 5, b: 12}))

    sess = tf.Session()
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    # Make sure to initialise tf global variables
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # Simple train API
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess.run(init) # reset values to incorrect defaults
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([W, b]))


if __name__ == '__main__':
    main()
