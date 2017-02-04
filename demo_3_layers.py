from numpy import exp, array, random, dot
import numpy as np

class NeuronLayer():
  """Init NeuroLayer"""
  def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
    self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) \
      - 1


class NeuralNetwork():
  """Init NeuralNetwork"""
  def __init__(self, layer1, layer2):
    self.layer1 = layer1
    self.layer2 = layer2
    
  # The Sigmoid function, which describes an S shaped curve.
  # We pass the weighted sum of the inputs through this function to
  # normalise them between 0 and 1.  
  def __sigmoid(self, x):
    return 1 / (1 + exp(-x))

  # The derivative of the Sigmoid function.
  # This is the gradient of the Sigmoid curve.
  # It indicates how confident we are about the existing weight.
  def __sigmoid_derivative(self, x):
    return x * (1- x)

  def think(self, inputs):
    output_from_layer_1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
    output_from_layer_2 = self.__sigmoid(dot(output_from_layer_1, self.layer2.synaptic_weights))
    return output_from_layer_1, output_from_layer_2

  # We train the neural network through a process of trial and error.
  # Adjusting the synaptic weights each time.
  def train(self, training_set_inputs, training_set_outputs, number_of_training_interations):
    for iteration in range(number_of_training_interations):
      # Pass the training set through our neural network
      output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)


      # Calculate the error for layer 2 (The difference between the desired output
      # and the predicted output).
      layer_2_error = training_set_outputs - output_from_layer_2
      layer_2_delta = layer_2_error * self.__sigmoid_derivative(output_from_layer_2)

      # Calculate the error for layer 1 (By looking at the weights in layer 1,
      # we can determine by how much layer 1 contributed to the error in layer 2).
      layer_1_error = layer_2_delta.dot(self.layer2.synaptic_weights.T)
      layer_1_delta = layer_1_error * self.__sigmoid_derivative(output_from_layer_1)

      layer1_adjustment = training_set_inputs.T.dot(layer_1_delta)
      layer2_adjustment = output_from_layer_1.T.dot(layer_2_delta)

      if (iteration % 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(layer_2_error))))

 


      self.layer1.synaptic_weights+= layer1_adjustment
      self.layer2.synaptic_weights+= layer2_adjustment

  def print_weights(self):
    print ("  Layer 1(4 neuron, each with 3 inputs): " )
    print (self.layer1.synaptic_weights)
    print ("  Layer 2(1 neuron, each with 4 inputs):     " )
    print (self.layer2.synaptic_weights)



if __name__ == '__main__':
  
  random.seed(1)

  layer1 = NeuronLayer(4, 3)
  layer2 = NeuronLayer(1, 4)

  neural_network = NeuralNetwork(layer1, layer2)

  print("Stage 1) Random starting synaptic weights: " )
  neural_network.print_weights()

  # The training set. We have 7 examples, each consisting of 3 input values
  # and 1 output value.
  training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
  training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

  neural_network.train(training_set_inputs, training_set_outputs, 60000)


  print("Stage 2) New synaptic weights after training: " )
  neural_network.print_weights()

  print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
  hidden_state, output = neural_network.think([1, 1, 0])
  print(output)
