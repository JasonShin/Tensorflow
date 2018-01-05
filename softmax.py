import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def main():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    print(type(mnist))
    print(mnist.train.images)
    print(mnist.test.num_examples)
    single = (mnist.train.images[1].reshape(28, 28))
    print(single)
    plt.imshow(single)
    print(single.min())
    print(single.max())
    pass


if __name__ == '__main__':
    main()
