import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    # x is just a placeholder that we want to tell TF to flatten each mnist
    # image into a 784-dimensional vector
    x = tf.placeholder(tf.float32, [None, 784])
    # We are going to learn W and b. It doesn't really matter what they
    # are initially
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Our softmax model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Training
    # In ML, we typically define what it means for a model to be bad. We call
    # this "cost" or "loss". It represents how far off our model is from our
    # desired outcome
    # nice function to determine loss of a model is called "cross-entropy"
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Cross entropy function defined
    # explanation
    # https://en.wikipedia.org/wiki/Softmax_function#Artificial_neural_networks
    # tf.log computes the logarithm of each element of y
    # then multiply it with _y
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y),
                       reduction_indices=[1]))



if __name__ == '__main__':
    main()