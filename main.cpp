#include "project2_a.h"

int main() {
  // Instantiate an activation function (same for all layers)
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Build the network: 2,3,3,1 neurons in the layers
  // Number of neurons in the input layer
  unsigned n_input = 2;

  // Storage for the non-input layers: number of neurons and activation function
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;

  // First hidden layer with 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Second hidden layer with 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Output layer with 1 neuron
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));

  // Create the NeuralNetwork instance
  NeuralNetwork network(n_input, non_input_layer);
  network.initialise_parameters(0.0, 1.0);

  // Create the NeuralNetwork instance
  NeuralNetwork network(n_input, non_input_layer);

  // Proceed with initializing parameters and testing
  return 0;
}