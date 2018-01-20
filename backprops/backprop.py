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


def main():
    network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
               [{'weights': [0.2550690257394217, 0.49543508709194095]},
                {'weights': [0.4494910647887381, 0.651592972722763]}]]
    row = [1, 0, None]
    # Forward prop example
    output = forward_propagate(network, row)
    print('checking stuff!', output)


if __name__ == '__main__':
    main()
