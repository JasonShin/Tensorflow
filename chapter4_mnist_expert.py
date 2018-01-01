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


if __name__ == '__main__':
    main()
