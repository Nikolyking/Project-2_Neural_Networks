
// Header guard to prevent multiple inclusions of the header file
#ifndef PROJECT2_A_H
#define PROJECT2_A_H

// Include necessary header files
#include "project2_a_basics.h"
#include "dense_linear_algebra.h"

// Include standard libraries
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>

using namespace BasicDenseLinearAlgebra;

// NeuralNetworkLayer class represents a single layer in the network
class NeuralNetworkLayer {
public:
  // Constructor
  NeuralNetworkLayer(unsigned n_input, unsigned n_neurons, ActivationFunction* activation_function);

  // Compute the output of the layer given the input
  void compute_output(const DoubleVector& input, DoubleVector& output) const;

  // Activation function pointer
  ActivationFunction* activation_function_pt;

  // Bias vector
  DoubleVector biases;

  // Weight matrix
  DoubleMatrix weights;

    // Getter functions
    unsigned get_n_input() const;
    unsigned get_n_neurons() const;

private:
  // Number of inputs to the layer (from the previous layer)
  unsigned n_input;

  // Number of neurons in the layer
  unsigned n_neurons;
};

// Getter for n_input
unsigned NeuralNetworkLayer::get_n_input() const {
  return n_input;
}

// Getter for n_neurons
unsigned NeuralNetworkLayer::get_n_neurons() const {
  return n_neurons;
}

// NeuralNetworkLayer constructor implementation
NeuralNetworkLayer::NeuralNetworkLayer(unsigned n_input_, unsigned n_neurons_, ActivationFunction* activation_function_)
  : n_input(n_input_), n_neurons(n_neurons_), activation_function_pt(activation_function_) {
  
  // Initialize biases to zero
  biases = DoubleVector(n_neurons);
  
  // Initialize weights to zero; dimensions: n_neurons x n_input
  weights = DoubleMatrix(n_neurons, n_input);
}

// Compute the output of the layer given the input
void NeuralNetworkLayer::compute_output(const DoubleVector& input, DoubleVector& output) const {
  // Ensure input size matches expected input size
  assert(input.n() == n_input);

  // Initialize output vector
  output.resize(n_neurons);

  // For each neuron, compute the weighted input and apply activation function
  for (unsigned i = 0; i < n_neurons; ++i) {
    double weighted_input = biases[i];

    // Compute weighted sum of inputs
    for (unsigned j = 0; j < n_input; ++j) {
      weighted_input += weights(i, j) * input[j];
    }

    // Apply activation function
    output[i] = activation_function_pt->sigma(weighted_input);
  }
}

// Define the NeuralNetwork class that inherits from NeuralNetworkBasis
class NeuralNetwork : public NeuralNetworkBasis {
public:
  // Constructor: Initializes the network structure
  NeuralNetwork(
    const unsigned& n_input,
    const std::vector<std::pair<unsigned, ActivationFunction*>>& non_input_layer);

  // Override pure virtual functions with dummy implementations
  
  // Feed-forward function
  void feed_forward(const DoubleVector& input, DoubleVector& output) const override;

  // Compute the cost for a single input-output pair
  double cost(const DoubleVector& input, const DoubleVector& target_output) const override;

  // Compute the cost over the entire training dataset
  double cost_for_training_data(
    const std::vector<std::pair<DoubleVector, DoubleVector>> training_data) const override;

  // Write network parameters to disk
  void write_parameters_to_disk(const std::string& filename) const override;

  // Read network parameters from disk
  void read_parameters_from_disk(const std::string& filename) override;

  // Train the neural network using stochastic gradient descent
  void train(
    const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data,
    const double& learning_rate,
    const double& tol_training,
    const unsigned& max_iter,
    const std::string& convergence_history_file_name = "") override;

  // Initialize weights and biases
  void initialise_parameters(const double& mean, const double& std_dev) override;

  // Initialize weights and biases for test
  void initialise_parameters_for_test(const double& value);

private:
  // Vector of layers in the network
  std::vector<NeuralNetworkLayer> layers;
};

// NeuralNetwork constructor implementation
NeuralNetwork::NeuralNetwork(
  const unsigned& n_input,
  const std::vector<std::pair<unsigned, ActivationFunction*>>& non_input_layer) {
  
  // Initialize the network layers
  unsigned previous_layer_size = n_input;

  for (const auto& layer_info : non_input_layer) {
    unsigned n_neurons = layer_info.first;
    ActivationFunction* activation_function = layer_info.second;

    // Create a new layer
    NeuralNetworkLayer layer(previous_layer_size, n_neurons, activation_function);

    // Add the layer to the network
    layers.push_back(layer);

    // Update previous_layer_size for the next layer
    previous_layer_size = n_neurons;
  }
}

// Feed-forward function implementation
void NeuralNetwork::feed_forward(const DoubleVector& input, DoubleVector& output) const {
  assert(!layers.empty());
  
  // Check that the input size matches the expected input size
  assert(input.n() == layers.front().get_n_input());

  DoubleVector layer_input = input;
  DoubleVector layer_output;

  // Process each layer sequentially
  for (const auto& layer : layers) {
    // Compute the output of the current layer
    layer.compute_output(layer_input, layer_output);

    // The output of the current layer becomes the input to the next layer
    layer_input = layer_output;
  }

  // The final output is the output from the last layer
  output = layer_output;
}

// Cost function for a single input-output pair
double NeuralNetwork::cost(const DoubleVector& input, const DoubleVector& target_output) const {
  std::cerr << "Warning: cost() not implemented yet." << std::endl;
  std::abort();
  return 0.0; // Added to prevent compiler warnings
}

// Cost function over the entire training dataset
double NeuralNetwork::cost_for_training_data(
    const std::vector<std::pair<DoubleVector, DoubleVector>> training_data) const {
  std::cerr << "Warning: cost_for_training_data() not implemented yet." << std::endl;
  std::abort();
  return 0.0; // Added to prevent compiler warnings
}

// Write network parameters to disk
void NeuralNetwork::write_parameters_to_disk(const std::string& filename) const {
  std::cerr << "Warning: write_parameters_to_disk() not implemented yet." << std::endl;
  std::abort();
}

// Read network parameters from disk
void NeuralNetwork::read_parameters_from_disk(const std::string& filename) {
  std::cerr << "Warning: read_parameters_from_disk() not implemented yet." << std::endl;
  std::abort();
}

// Train the neural network
void NeuralNetwork::train(
  const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data,
  const double& learning_rate,
  const double& tol_training,
  const unsigned& max_iter,
  const std::string& convergence_history_file_name) {
  std::cerr << "Warning: train() not implemented yet." << std::endl;
  std::abort();
}

// Initialize weights and biases with random values
void NeuralNetwork::initialise_parameters(const double& mean, const double& std_dev) {
  // Create a random number generator and normal distribution
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean, std_dev);

  // Iterate over each layer in the network
  for (auto& layer : layers) {
    // Initialize biases
    for (unsigned i = 0; i < layer.biases.n(); ++i) {
      layer.biases[i] = distribution(generator);
    }

    // Initialize weights
    for (unsigned i = 0; i < layer.weights.n(); ++i) {
      for (unsigned j = 0; j < layer.weights.m(); ++j) {
        layer.weights(i, j) = distribution(generator);
      }
    }
  }
}

// Initialize weights and biases with a specific value for testing
void NeuralNetwork::initialise_parameters_for_test(const double& value) {
  // Iterate over each layer in the network
  for (auto& layer : layers) {
    // Initialize biases
    for (unsigned i = 0; i < layer.biases.n(); ++i) {
      layer.biases[i] = value;
    }

    // Initialize weights
    for (unsigned i = 0; i < layer.weights.n(); ++i) {
      for (unsigned j = 0; j < layer.weights.m(); ++j) {
        layer.weights(i, j) = value;
      }
    }
  }
}

#endif // PROJECT2_A_H