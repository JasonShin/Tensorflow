import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # INIT WEIGHTS
    def init_weights(shape):
        # ref to truncated normal https://www.tensorflow.org/api_docs/python/tf/truncated_normal
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)

    # INIT BIAS
    def init_bias(shape):
        # ref: https://www.tensorflow.org/versions/master/api_docs/python/tf/constant
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(x, W):
        # x --> [batch, H, W, Channels]
        # W --> [filter H, filter W, Channels]
        # ref https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # POOLING
    def max_pool_2by2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def convolutional_layer(input_x, shape):
        W = init_weights(shape)
        b = init_bias([shape[3]])
        return tf.nn.relu(conv2d(input_x, W) + b)

    # NORMAL (FC)
    def normal_full_layer(input_layer, size):
        input_size = int(input_layer.getshape()[1])
        W = init_weights([input_size, size])
        b = init_bias([size])
        return tf.matmul(input_layer, W) + b

    # PLACEHOLDERS
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    # Layers
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convo1
    convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
    convo_1_pooling = max_pool_2by2(convo_1)

    # convo2
    convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
    convo_2_pooling = max_pool_2by2(convo_2)

    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])
    full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

if __name__ == '__main__':
    main()
