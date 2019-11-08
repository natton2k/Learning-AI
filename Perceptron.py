import numpy


def hardlims(x):
    if x >= 0:
        return 1
    else:
        return 0

class Neuron:
    def __init__(self, input_number, function, order):
        self.input_number = input_number
        self.weight = numpy.zeros((1, self.input_number))
        self.function = function
        self.bias = 0
        self.order = order

    def __call__(self, sample):
        sample_input = numpy.array(sample[0]).reshape(self.input_number, 1)
        sample_result = sample[1][self.order]
        result = self.function(float(numpy.dot(self.weight, sample_input)) + self.bias)
        loss = sample_result - result
        self.update(loss, sample_input)

    def update(self, loss, sample_input):
        learn = numpy.multiply(sample_input.reshape(1, self.input_number), numpy.full((1, self.input_number), loss))
        self.weight = numpy.add(self.weight, learn)
        self.bias = (self.bias + loss)/numpy.max(self.weight)
        self.weight = numpy.divide(self.weight,numpy.full((1,self.input_number),numpy.max(self.weight)))




class Network1:  # with neuron
    def __init__(self, neuron_number, input_number, function):
        self.neurons = [Neuron(input_number, function, i) for i in range(neuron_number)]

    def __call__(self, samples):
        for sample in samples:
            for neuron in self.neurons:
                neuron(sample)
    def weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weight)
        return weights
    def biases(self):
        biases = []
        for neuron in self.neurons:
            biases.append(neuron.bias)
        return biases

network = Network1(1,3, hardlims)
sample = [
    ([-1,-1,-1],[-1]),
    ([1,1,2],[1])
]
for _ in range(10000):
    network(sample)
print(network.weights())
print(network.biases())

a = numpy.zeros((1,5))

