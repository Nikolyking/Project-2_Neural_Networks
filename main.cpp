#include "project2_a.h"

int main() {

  std::string training_data_file = "project_training_data.dat";
  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
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
  network.initialise_parameters(0.0, 0.1);
  network.read_training_data(training_data_file, training_data);

  // Train the network
  double learning_rate = 0.001;
  double tol_training = 0.01;
  unsigned max_iter = 100000;
  std::string convergence_history_file = "convergence_history.txt";

  network.train(training_data, learning_rate, tol_training, max_iter, convergence_history_file);

  // Evaluate the network on the training data
  double total_cost = network.cost_for_training_data(training_data);
  std::cout << "Total cost for training data after training: " << total_cost << std::endl;

  // Save the trained parameters to a file
  network.write_parameters_to_disk("trained_parameters.dat");


  // Clean up
  delete activation_function_pt;

  // Proceed with initializing parameters and testing
  return 0;
}