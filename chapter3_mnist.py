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

    # automatic backpropogation
    # http://colah.github.io/posts/2015-08-Backprop/
    # Grapdient Descent optimzier trainer with .5 learning rate
    # TF simply shifts each variable a little bit in the direction
    # that reduces the cost
    # However this is one the many optimizers https://www.tensorflow.org/api_guides/python/train#Optimizers
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Launch the model in an InteractiveSession
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Run the training step for 1000 times!
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evaluating model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # above gives us list of booleans
    # [True, False, True, True] would become [1,0,1,1] which would become 0.75
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Checking accuracy x = mnist test images')
    print('y_ mnist test labels')
    print(sess.run(accuracy,
                   feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main()