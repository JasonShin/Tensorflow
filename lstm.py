import numpy as np


class RecurrentNeuralNetwork:
    def __init__(self, xs, ys, rl, eo, lr):
        # initial input (first word)
        self.x = np.zeros(xs)

        # input size
        self.xs = xs

        # expected output (next word)
        self.y = np.zeros(ys)

        # output size
        self.ys = ys

        # weight matrix for interpreting results from LSTM cell (num words x num words matrix)
        self.w = np.random.random((ys, ys))

        # matrix used in RSMprop
        self.G = np.zeros_like(self.w)

        # length of the recurrent network - number of recurrences i.e. num of words
        self.rl = rl

        # array for storing inputs
        self.ia = np.zeros((rl + 1, xs))

        # array for storing outputs
        self.ca = np.zeros((rl + 1, ys))

        # array for storing cell states
        self.ca = np.zeros((rl + 1, ys))

        # array for storing outputs
        self.oa = np.zeros((rl + 1, ys))

        # array for storing hidden states
        self.ha = np.zeros((rl + 1, ys))

        # forget gate
        self.af = np.zeros((rl + 1, ys))

        # input gate
        self.ai = np.zeros((rl + 1, ys))




def main():
    print('main')


if __name__ == '__main__':
    main()


