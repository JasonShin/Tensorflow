from tensorflow.examples.tutorials.mnist import input_data

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('checking main')


if __name__ == '__main__':
    main()
