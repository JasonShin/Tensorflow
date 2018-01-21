from math import exp


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        print('layer: ', layer)
        for neuron in layer:
            print('checking neuron', neuron)
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


# Backpopagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def main():
    # Forward prop example
    network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
               [{'weights': [0.2550690257394217, 0.49543508709194095]},
                {'weights': [0.4494910647887381, 0.651592972722763]}]]
    row = [1, 0, None]
    output = forward_propagate(network, row)

    # test backpropagation of error
    network = [
        [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
         {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    expected = [0, 1]
    backward_propagate_error(network, expected)
    print('Finished backpropagating errors')
    for layer in network:
        print(layer)


if __name__ == '__main__':
    main()
