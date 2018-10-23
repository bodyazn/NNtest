import math, random


class Neuron:

    learning_rate = 0.2  # not sure what is best value for current learning set and NN topology
    alpha_to_avoid_local_minimum = 0.3  # actually it is momentum

    def __init__(self, num_of_inputs, layer_name=""):
        self.layer_name = layer_name

        self.weights = [random.random() * 2.0 - 1 for i in range(0, num_of_inputs)]
        self.delta_weights = [0 for i in range(0, num_of_inputs)]

        self.input_values = []

        self.err = 0.0
        self.err_count_collected_from_the_right = 0

        self.input_neurons = None
        self.output_neurons = None

    def get_output(self, inputs):
        s = 0
        for i in range(len(inputs)):
            s += inputs[i] * self.weights[i]
        return Neuron.nonlin(s)

    def nonlin(x, deriv=False, deriv_based_on_sigmoid_output = True):
        if deriv:
            if deriv_based_on_sigmoid_output:
                return x * (1 - x)
            else:
                return Neuron.nonlin(x) * (1 - Neuron.nonlin(x))  # f(x) * (1 - f(x))
        return 1 / (1 + math.exp(-x))

    def add_input_neuron(self, input_neuron):
        if self.input_neurons == None:
            self.input_neurons = []
        if input_neuron not in self.input_neurons:
            self.input_neurons.append(input_neuron)
            input_neuron.add_output_neuron(self)

    def add_output_neuron(self, output_neuron):
        if self.output_neurons == None:
            self.output_neurons = []
        if output_neuron not in self.output_neurons:
            self.output_neurons.append(output_neuron)
            output_neuron.add_input_neuron(self)

    def register_inputs_neurons(self, neurons):
        self.input_neurons = neurons
        for i_n in neurons:
            i_n.add_output_neuron(self)

    def register_output_neurons(self, neurons):
        self.output_neurons = neurons
        for o_n in neurons:
            o_n.add_input_neuron(self)

    def out_to_input_call(self):
        #print("out_to_input_call in layer: {}".format(self.layer_name))

        if self.input_neurons != None:
            self.input_values = [i_n.out_to_input_call() for i_n in self.input_neurons]
        return self.get_output(self.input_values)

    def back_propagation(self, expected, layer_on_the_right_delta=None):
        output = self.out_to_input_call()

        if layer_on_the_right_delta is None:
            self.err = expected - output
        else:
            self.err += layer_on_the_right_delta
            self.err_count_collected_from_the_right += 1
            if self.err_count_collected_from_the_right < len(self.output_neurons):
                return  # continue collecting errors from the 'right side'


        # the change in weight for a single connection will be

        # δweight= η x gradient x output of connected neuron + α x previous δweight
        # here eta (η) is the learning rate which will decide how fast the network will update its weight and
        # alpha (α) is the momentum rate which will give the weights a momentum so that it will continue
        # to move in a particular direction ignoring small fluctuations.
        # which basically helps the neuron to avoid local minima

        gradient = self.err * Neuron.nonlin(output, True)  # f(x) * (1 - f(x))

        for i in range(len(self.weights)):

            d_weight = Neuron.learning_rate * gradient * self.input_values[i] \
                       + Neuron.alpha_to_avoid_local_minimum * self.delta_weights[i]
            self.weights[i] += d_weight
            self.delta_weights[i] = d_weight

            self.err = 0.0
            self.err_count_collected_from_the_right = 0  # reset when train 'tick' is done

        if self.input_neurons:
            for i in range(len(self.input_neurons)):
                i_n = self.input_neurons[i]
                i_n.back_propagation(None, self.weights[i] * gradient)  # sending error to the left



class NLayer:
    def __init__(self, layer_name, num_of_neurons, inputs_per_neuron = 1, input_layer = None, output_layer = None):
        self.name = layer_name
        if input_layer != None:
            inputs_per_neuron = len(input_layer.neurons)
        self.neurons = [Neuron(inputs_per_neuron, layer_name) for i in range(num_of_neurons)]
        for n in self.neurons:
            if input_layer != None:
                n.register_inputs_neurons(input_layer.neurons)
            if output_layer != None:
                n.register_output_neurons(output_layer.neurons)

    def register_output_layer(self, output_layer):
        for n in self.neurons:
            n.register_output_neurons(output_layer.neurons)


    def get_info(self):
        return (self.name, self.neurons)


'''
input_layer = NLayer("input", 4)
output_layer = NLayer("output", 3, 4)
layer1 = NLayer("hidden 1", 4, 4, input_layer, output_layer)
#layer2 = NLayer("hidden 2", 4, 4)
#layer1 = NLayer("hidden 1", 4, 1, input_layer, layer2)
#layer2.register_output_layer(output_layer)
'''
input_layer = NLayer("input", 4)
output_layer = NLayer("output", 3, 4)
layer1 = NLayer("hidden 1", 4, 4, input_layer)
layer2 = NLayer("hidden 2", 4, 4, layer1, output_layer)
#layer1 = NLayer("hidden 1", 4, 1, input_layer, layer2)
#layer2.register_output_layer(output_layer)


learning_set = [
    [0, 0, 0, 0],  # 0
    [0, 0, 0, 1],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 1, 1],  # 3
    [0, 1, 0, 0],  # 4
    [0, 1, 0, 1],  # 5
    [0, 1, 1, 0],  # 6
    [0, 1, 1, 1],  # 7
    [1, 0, 0, 0],  # 8
    [1, 0, 0, 1],  # 9
    [1, 0, 1, 0],  # 10
    [1, 0, 1, 1],  # 11
    [1, 1, 0, 0],  # 12
    [1, 1, 0, 1],  # 13
    [1, 1, 1, 0],  # 14
    [1, 1, 1, 1]  # 15
]
'''
correct_responses = [
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
]

'''
correct_responses = [
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]


def nn_print_output(ls, output_layer, correct_response):
    float_outputs = [round(x.out_to_input_call(), 3) for x in output_layer.neurons]
    int_outputs = [int(x >= 0.5) for x in float_outputs]
    is_ok = (int_outputs[0] == correct_response[0] and
             int_outputs[1] == correct_response[1] and
             int_outputs[2] == correct_response[2])

    print(ls,
          "\t", int_outputs[0], int_outputs[1], int_outputs[2],
          "\texpct: ", correct_response,
          "\t", float_outputs[0],
          "\t", float_outputs[1],
          "\t", float_outputs[2],
          "\tok" * is_ok)

    return is_ok

for epoch in range(30000):
    if epoch % 100 == 0:
        print(epoch, end="... ")
    if epoch % 1000 == 0:
        print()

    ok_count = 0
    for ls_i in range(len(learning_set)):
        #print(ls)
        #print(input_layer.neurons)
        ls = learning_set[ls_i]
        for i in range(len(ls)):
            input_layer.neurons[i].input_values = [ls[i]]
        output_layer.neurons[0].back_propagation(correct_responses[ls_i][0])
        output_layer.neurons[1].back_propagation(correct_responses[ls_i][1])
        output_layer.neurons[2].back_propagation(correct_responses[ls_i][2])

        if epoch % 1000 == 0:
            ok_count += nn_print_output(ls, output_layer, correct_responses[ls_i])
            if ok_count == len(learning_set):
                print("\n It seems enough for now ;) \n Epochs done:", epoch)
                break
    else:
        continue
    break


#  Final grand total ;)
print("\n\t *** Grand Total Output ***")
ok_count = 0
for l in range(len(learning_set)):
    ls = learning_set[l]

    for i in range(len(ls)):
        input_layer.neurons[i].input_values = [ls[i]]

    ok_count += nn_print_output(ls, output_layer, correct_responses[l])

print(f"OK: {ok_count} ({len(correct_responses)})")
