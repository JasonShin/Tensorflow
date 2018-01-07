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

        # cell state
        self.ac = np.zeros((rl + 1, ys))

        # output gate
        self.ao = np.zeros((rl + 1, ys))

        # array of expected output values
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))

        # declare LSTM cell (input, output, amount of recurrence, Learning rate)
        self.LSTM = LSTM(xs, ys, rl, lr)

    # activation function. simple nonlinearity, convert nums into probabilities between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # the derivative of the sigmoid function.
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forwardProp(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x))
            cs, hs, f, c, o = self.LSTM.forwardProp()
            # store computed cell state
            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp



def main():
    print('main')


if __name__ == '__main__':
    main()


