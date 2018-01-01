import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # mnist dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # interactive session
    sess = tf.InteractiveSession()
    # x input and y output placeholders
    # x = flattened 28 by 28 pixel dimensionality of mnist images
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # y = each row is one-hot 10 dimensional vector
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # define weight and bias
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.global_variables_initializer())

    # declare softmax model
    y = tf.matmul(x, W) + b

    # declare loss function
    # loss indicate how bad our model performs
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    )

    # train the model
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # apply gradient descent to each training iteration
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # tf.argmax(y, 1) = most likely result of each input
    # tf.argmax(y_, 1) = actual result
    # then equal if they are equal
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # To determine what fraction are correct, we cast to floating point
    # numbers and then take the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # finally test the accuracy
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # Convolutional layer 1
    # 5,5 => computing 32 features for each 5x5
    # 1 => number of input channel
    # 32 => number of output channel
    W_conv1 = weight_variable([5, 5, 1, 32])
    # bias vector with a component for each output channel
    b_conv1 = bias_variable([32])

    # Reshape x to 4d vector
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convolve x image with weight tensor, add the bias, apply relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # reduce the image size to 14 x 14
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolutional layer 2
    # To build a deep neural network, we must stack multiple layers of them
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    # Add a layer with 1024 neurons to allow processing on an entire image
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    # We will apply dropout to avoid overfitting
    keep_prop = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)

    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train model
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prop: 1.0
                })
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={
                x: batch[0], y_: batch[1], keep_prop: 0.5
            })

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# generating initial weight with small amount to prevent dead neurons
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    main()
