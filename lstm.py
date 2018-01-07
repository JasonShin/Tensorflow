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
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i - 1]
        return self.oa

    def backProp(self):
        # update out weight matrices (Both in our Re
        totalError = 0
        # initialize matrices for gradient updates
        dfcs = np.zeros(self.ys)
        # hidden state
        dfhs = np.zeros(self.ys)
        # weight matrix
        tu = np.zeros((self.ys, self.ys))
        # forget gate
        tfu = np.zeros((self. ys, self.xs + self.ys))
        # input gate
        tiu = np.zeros((self.ys, self.xs + self.ys))
        # cell unit
        tcu = np.zeros((self.ys, self.xs + self.ys))
        # output gate
        tou = np.zeros((self.ys, self.xs + self.ys))
        # loop backwards through recurrences
        for i in range(self.rl, -1, -1):
            # error = calculatedOutput - expectedOutput
            error = self.oa[i] - self.eo[i]
            # calculate update for weight matrix
            # (error * derivative of the output) * hidden state
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            # Time to propagate error back to exit of LSTM cell
            # 1. error * RNN weight matrix
            error = np.dot(error, self.w)
            # 2. set input values of LSTM cell for recurrence i (horizontal stack of arrays, hidden + input)
            self.LSTM.x = np.hstack((self.ha[i - 1], self.ia[i]))
            # 3. set cell state of LSTM cell for recurrence i (pre-updates)
            self.LSTM.cs = self.ca[i]
            # Finally, call the LSTM cell's backprop, retreive gradient updates
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(
                error, self.ca[i - 1], self.af[i], self.ai[i], self.ac[i], self.ao[i], dfcs, dfhs
            )
            # calculate total error (not necessary, used to measure training progress)
            totalError += np.sum(error)
            # accumulate all gradient updates
            # forget gate
            tfu += fu
            # input gate
            tiu += iu
            # cell state
            tcu += cu
            # output gate
            tou += ou
        # update LSTM matrices with average of accumulated gradient updates
        self.LSTM.update(tfu / self.rl, tiu / self.rl, tcu / self.rl, tou / self.rl)
        self.update(tu / self.rl)
        return totalError

    def update(self, u):
        # vanilla implementation of RMSprop
        self.G = 0.9 * self.G + 0.1 * u**2
        self.w -= self.lr / np.sqrt(self.G + 1e-8) * u

    # this is where we generate some sample text after having fully trained our model


def main():
    print('main')


if __name__ == '__main__':
    main()


