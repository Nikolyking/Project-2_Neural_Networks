
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
#include <algorithm>

// Namespace for linear algebra
using namespace BasicDenseLinearAlgebra;

// Class representing a single layer
class NeuralNetworkLayer {
public:
  // Constructor
  NeuralNetworkLayer(unsigned n_input, unsigned n_neurons, ActivationFunction *activation_function);

  // Compute the output of the layer given the input
  void compute_output(const DoubleVector &input, DoubleVector &output) const;

  // Activation function pointer
  ActivationFunction *activation_function_pt;

  // Bias vector
  DoubleVector biases;

  // Weight matrix
  DoubleMatrix weights;

  // Getter functions
  unsigned get_n_input() const;
  unsigned get_n_neurons() const;

private:
  // Number of inputs to the layer
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
// "_" sign used to distinguish between class variables and other variables 
NeuralNetworkLayer::NeuralNetworkLayer(unsigned n_input_, unsigned n_neurons_, ActivationFunction *activation_function_)
    : n_input(n_input_), n_neurons(n_neurons_), activation_function_pt(activation_function_) {

  // Initialize biases to zero
  biases = DoubleVector(n_neurons);

  // Initialize weights to zero; dimensions: n_neurons x n_input
  weights = DoubleMatrix(n_neurons, n_input);
}

// Compute the output of the layer given the input
void NeuralNetworkLayer::compute_output(const DoubleVector &input, DoubleVector &output) const {
  // Ensure input size matches expected input size
  assert(input.n() == n_input);

  // Output vector initialization
  output.resize(n_neurons);

  // For each neuron, compute the weighted input and apply activation function
  for (unsigned i = 0; i < n_neurons; ++i) {
    double weighted_input = biases[i];

    // Compute weighted sum of inputs
    for (unsigned j = 0; j < n_input; ++j) {
      weighted_input += weights(i, j) * input[j];
    }

    // Apply activation function
    // "->" sign used to access sigma through a pointer 
    output[i] = activation_function_pt->sigma(weighted_input);
  }
}

// Define the NeuralNetwork class that inherits from NeuralNetworkBasis
class NeuralNetwork : public NeuralNetworkBasis {
public:
  // Constructor: Initializes the network structure
  NeuralNetwork(
      const unsigned &n_input,
      const std::vector<std::pair<unsigned, ActivationFunction *>> &non_input_layer);

  // Override pure virtual functions with dummy implementations

  // Feed-forward function
  void feed_forward(const DoubleVector &input, DoubleVector &output) const override;

  // Compute the cost for a single input-output pair
  double cost(const DoubleVector &input, const DoubleVector &target_output) const override;

  // Compute the cost over the entire training dataset
  double cost_for_training_data(
      const std::vector<std::pair<DoubleVector, DoubleVector>> training_data) const override;

  // Write network parameters to disk
  void write_parameters_to_disk(const std::string &filename) const override;

  void read_training_data(const std::string &filename, std::vector<std::pair<DoubleVector, DoubleVector>> &training_data) const;

  // Read network parameters from disk
  void read_parameters_from_disk(const std::string &filename) override;

  // Train the neural network using stochastic gradient descent
  void train(
      const std::vector<std::pair<DoubleVector, DoubleVector>> &training_data,
      const double &learning_rate,
      const double &tol_training,
      const unsigned &max_iter,
      const std::string &convergence_history_file_name = "") override;

  // Initialize weights and biases
  void initialise_parameters(const double &mean, const double &std_dev) override;

  // Initialize weights and biases for test
  void initialise_parameters_for_test(const double &value);

private:
  // Vector of layers in the network
  std::vector<NeuralNetworkLayer> layers;

  // Static random number generator with fixed seed
  static std::mt19937 random_generator;

  void feed_forward_with_intermediate_results(
      const DoubleVector &input,
      std::vector<DoubleVector> &activations,
      std::vector<DoubleVector> &weighted_inputs) const;

  DoubleVector compute_output_layer_delta(
      const DoubleVector &output_activations,
      const DoubleVector &target_output,
      const DoubleVector &weighted_input,
      const NeuralNetworkLayer &output_layer) const;

  std::vector<DoubleVector> backpropagate_error(
      const DoubleVector &output_delta,
      const std::vector<DoubleVector> &weighted_inputs,
      const std::vector<DoubleVector> &activations) const;

  void update_weights_and_biases(
      const std::vector<DoubleVector> &deltas,
      const std::vector<DoubleVector> &activations,
      const double &learning_rate);

  void compute_gradients_finite_diff(
      const DoubleVector &input,
      const DoubleVector &target_output,
      std::vector<DoubleMatrix> &weight_gradients,
      std::vector<DoubleVector> &bias_gradients,
      const double eps);

  void validate_gradients(
      const DoubleVector &input,
      const DoubleVector &target_output);
};

// NeuralNetwork constructor implementation
NeuralNetwork::NeuralNetwork(
    const unsigned &n_input,
    const std::vector<std::pair<unsigned, ActivationFunction *>> &non_input_layer)
{

  // Initialize the network layers
  unsigned previous_layer_size = n_input;

  for (const auto &layer_info : non_input_layer) {
    unsigned n_neurons = layer_info.first;
    ActivationFunction *activation_function = layer_info.second;

    // Create a new layer
    NeuralNetworkLayer layer(previous_layer_size, n_neurons, activation_function);

    // Add the layer to the network
    layers.push_back(layer);

    // Update previous_layer_size for the next layer
    previous_layer_size = n_neurons;
  }
}

// Feed-forward function implementation
void NeuralNetwork::feed_forward(const DoubleVector &input, DoubleVector &output) const {
  assert(!layers.empty());

  // Check that the input size matches the expected input size
  assert(input.n() == layers.front().get_n_input());

  DoubleVector layer_input = input;
  DoubleVector layer_output;

  // Process each layer sequentially
  for (const auto &layer : layers) {
    // Compute the weighted input
    DoubleVector weighted_input(layer.get_n_neurons());
    for (unsigned j = 0; j < layer.get_n_neurons(); ++j) {
      weighted_input[j] = layer.biases[j];
      for (unsigned k = 0; k < layer.get_n_input(); ++k) {
        weighted_input[j] += layer.weights(j, k) * layer_input[k];
      }
    }

    // Apply the activation function a[l] = σ(z[l])
    layer_output.resize(layer.get_n_neurons());
    for (unsigned j = 0; j < layer.get_n_neurons(); ++j) {
      layer_output[j] = layer.activation_function_pt->sigma(weighted_input[j]);
    }

    // Update layer_input for the next layer
    layer_input = layer_output;
  }
  // The final output is the output from the last layer
  output = layer_output;
}

// Cost function for a single input-output pair
double NeuralNetwork::cost(const DoubleVector &input, const DoubleVector &target_output) const {
  // Perform feed-forward to get the network's output
  DoubleVector network_output;
  feed_forward(input, network_output);

  // Compute the cost (Mean Squared Error)
  double total_cost = 0.0;
  for (unsigned i = 0; i < target_output.n(); ++i) {
    double diff = network_output[i] - target_output[i];
    total_cost += 0.5 * diff * diff;
  }

  return total_cost;
}

// Cost function over the entire training dataset
double NeuralNetwork::cost_for_training_data(
    const std::vector<std::pair<DoubleVector, DoubleVector>> training_data) const {
  double total_cost = 0.0;

  // Iterate over all input-target pairs in the training set
  for (const auto &data_pair : training_data) {
    const DoubleVector &input = data_pair.first;
    const DoubleVector &target_output = data_pair.second;

    // Compute the cost for the current pair and add it to the total cost
    total_cost += cost(input, target_output);
  }

  // Compute the average cost
  total_cost /= training_data.size();

  return total_cost;
}

// Write network parameters to disk
void NeuralNetwork::write_parameters_to_disk(const std::string &filename) const {
  std::ofstream outfile(filename.c_str());
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to open file for writing: " + filename);
  }

  for (const auto &layer : layers) {
    // Write the name of the activation function
    outfile << layer.activation_function_pt->name() << std::endl;

    // Write the number of inputs to the layer
    outfile << layer.get_n_input() << std::endl;

    // Write the number of neurons in the current layer
    outfile << layer.get_n_neurons() << std::endl;

    // Write the bias vector
    for (unsigned i = 0; i < layer.biases.n(); ++i) {
      outfile << i << " " << layer.biases[i] << std::endl;
    }

    // Write the weight matrix
    for (unsigned i = 0; i < layer.weights.n(); ++i) {
      for (unsigned j = 0; j < layer.weights.m(); ++j) {
        outfile << i << " " << j << " " << layer.weights(i, j) << std::endl;
      }
    }
  }

  outfile.close();
}

void NeuralNetwork::read_training_data(const std::string &filename, std::vector<std::pair<DoubleVector, DoubleVector>> &training_data) const {
  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }

  unsigned n_training_sets, n_input, n_output;
  infile >> n_training_sets >> n_input >> n_output;

  for (unsigned i = 0; i < n_training_sets; ++i) {
    DoubleVector input(n_input);
    DoubleVector output(n_output);

    for (unsigned j = 0; j < n_input; ++j) {
      infile >> input[j];
    }

    for (unsigned j = 0; j < n_output; ++j) {
      infile >> output[j];
    }

    training_data.push_back(std::make_pair(input, output));
  }

  std::string end_of_file;
  infile >> end_of_file;
  if (end_of_file != "end_of_file") {
    throw std::runtime_error("Training data file format error: missing 'end_of_file' marker.");
  }

  infile.close();
}

// Read network parameters from disk
void NeuralNetwork::read_parameters_from_disk(const std::string &filename) {
  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }

  for (auto &layer : layers) {
    // Read and verify the activation function name
    std::string activation_function_name;
    infile >> activation_function_name;
    if (activation_function_name != layer.activation_function_pt->name()) {
      throw std::runtime_error("Activation function name mismatch: expected " +
                               layer.activation_function_pt->name() + ", got " + activation_function_name);
    }

    // Read and verify the number of inputs to the layer
    unsigned n_input;
    infile >> n_input;
    if (n_input != layer.get_n_input()) {
      throw std::runtime_error("Number of inputs mismatch: expected " +
                               std::to_string(layer.get_n_input()) + ", got " + std::to_string(n_input));
    }

    // Read and verify the number of neurons in the current layer
    unsigned n_neurons;
    infile >> n_neurons;
    if (n_neurons != layer.get_n_neurons()) {
      throw std::runtime_error("Number of neurons mismatch: expected " +
                               std::to_string(layer.get_n_neurons()) + ", got " + std::to_string(n_neurons));
    }

    // Read the bias vector
    for (unsigned i = 0; i < layer.biases.n(); ++i) {
      unsigned index;
      double value;
      infile >> index >> value;
      if (index != i) {
        throw std::runtime_error("Bias index mismatch: expected " + std::to_string(i) + ", got " + std::to_string(index));
      }
      layer.biases[i] = value;
    }

    // Read the weight matrix
    for (unsigned i = 0; i < layer.weights.n(); ++i) {
      for (unsigned j = 0; j < layer.weights.m(); ++j) {
        unsigned row, col;
        double value;
        infile >> row >> col >> value;
        if (row != i || col != j) {
          throw std::runtime_error("Weight index mismatch: expected (" + std::to_string(i) + ", " + std::to_string(j) +
                                   "), got (" + std::to_string(row) + ", " + std::to_string(col) + ")");
        }
        layer.weights(i, j) = value;
      }
    }
  }

  infile.close();
}

void NeuralNetwork::feed_forward_with_intermediate_results(
    const DoubleVector &input,
    std::vector<DoubleVector> &activations,
    std::vector<DoubleVector> &weighted_inputs) const {

  DoubleVector layer_input = input;
  activations.push_back(layer_input);

  for (const auto &layer : layers) {
    DoubleVector weighted_input(layer.get_n_neurons());
    for (unsigned j = 0; j < layer.get_n_neurons(); ++j) {
      weighted_input[j] = layer.biases[j];
      for (unsigned k = 0; k < layer.get_n_input(); ++k) {
        weighted_input[j] += layer.weights(j, k) * layer_input[k];
      }
    }
    weighted_inputs.push_back(weighted_input);

    DoubleVector layer_output(layer.get_n_neurons());
    for (unsigned j = 0; j < layer.get_n_neurons(); ++j) {
      layer_output[j] = layer.activation_function_pt->sigma(weighted_input[j]);
    }
    activations.push_back(layer_output);

    layer_input = layer_output;
  }
}

DoubleVector NeuralNetwork::compute_output_layer_delta(
    const DoubleVector &output_activations,
    const DoubleVector &target_output,
    const DoubleVector &weighted_input,
    const NeuralNetworkLayer &output_layer) const {

  DoubleVector delta(output_activations.n());
  for (unsigned j = 0; j < output_activations.n(); ++j) {
    double sigma_prime = output_layer.activation_function_pt->dsigma(weighted_input[j]);
    delta[j] = sigma_prime * (output_activations[j] - target_output[j]);
  }
  return delta;
}

std::vector<DoubleVector> NeuralNetwork::backpropagate_error(
    const DoubleVector &output_delta,
    const std::vector<DoubleVector> &weighted_inputs,
    const std::vector<DoubleVector> &activations) const {

  std::vector<DoubleVector> deltas(layers.size());
  deltas.back() = output_delta;

  for (int l = layers.size() - 2; l >= 0; --l) {
    const auto &layer = layers[l];
    const auto &next_layer = layers[l + 1];
    DoubleVector delta(layer.get_n_neurons());

    for (unsigned j = 0; j < layer.get_n_neurons(); ++j) {
      double sigma_prime = layer.activation_function_pt->dsigma(weighted_inputs[l][j]);
      double sum = 0.0;
      for (unsigned k = 0; k < next_layer.get_n_neurons(); ++k) {
        sum += next_layer.weights(k, j) * deltas[l + 1][k];
      }
      delta[j] = sigma_prime * sum;
    }
    deltas[l] = delta;
  }
  return deltas;
}

void NeuralNetwork::update_weights_and_biases(
    const std::vector<DoubleVector> &deltas,
    const std::vector<DoubleVector> &activations,
    const double &learning_rate) {

  for (unsigned l = 0; l < layers.size(); ++l) {
    auto &layer = layers[l];
    const auto &delta = deltas[l];
    const auto &activation = activations[l];

    for (unsigned j = 0; j < layer.get_n_neurons(); ++j) {
      layer.biases[j] -= learning_rate * delta[j];
      for (unsigned k = 0; k < layer.get_n_input(); ++k) {
        layer.weights(j, k) -= learning_rate * delta[j] * activation[k];
      }
    }
  }
}

void NeuralNetwork::compute_gradients_finite_diff(
    const DoubleVector &input,
    const DoubleVector &target_output,
    std::vector<DoubleMatrix> &weight_gradients,
    std::vector<DoubleVector> &bias_gradients,
    const double eps) {

  // Initialize gradient storage
  weight_gradients.resize(layers.size());
  bias_gradients.resize(layers.size());

  // Store original cost
  double original_cost = cost(input, target_output);

  // Compute gradients for each layer
  for (size_t l = 0; l < layers.size(); ++l) {
    // Bias gradients
    bias_gradients[l] = DoubleVector(layers[l].get_n_neurons());
    for (unsigned j = 0; j < layers[l].get_n_neurons(); ++j) {
      double original_bias = layers[l].biases[j];
      layers[l].biases[j] += eps;
      double perturbed_cost = cost(input, target_output);
      layers[l].biases[j] = original_bias;
      bias_gradients[l][j] = (perturbed_cost - original_cost) / eps;
    }

    // Weight gradients
    weight_gradients[l] = DoubleMatrix(layers[l].get_n_neurons(), layers[l].get_n_input());
    for (unsigned j = 0; j < layers[l].get_n_neurons(); ++j) {
      for (unsigned k = 0; k < layers[l].get_n_input(); ++k) {
        double original_weight = layers[l].weights(j, k);
        layers[l].weights(j, k) += eps;
        double perturbed_cost = cost(input, target_output);
        layers[l].weights(j, k) = original_weight;
        weight_gradients[l](j, k) = (perturbed_cost - original_cost) / eps;
      }
    }
  }
}

// Compare backprop and finite difference gradients
void NeuralNetwork::validate_gradients(
    const DoubleVector &input,
    const DoubleVector &target_output) {

  // Compute gradients using finite differences
  std::vector<DoubleMatrix> fd_weight_gradients;
  std::vector<DoubleVector> fd_bias_gradients;
  compute_gradients_finite_diff(input, target_output, fd_weight_gradients, fd_bias_gradients, 1e8);

  // Compute gradients using backpropagation
  std::vector<DoubleVector> activations;
  std::vector<DoubleVector> weighted_inputs;
  feed_forward_with_intermediate_results(input, activations, weighted_inputs);

  DoubleVector delta = compute_output_layer_delta(
      activations.back(), target_output, weighted_inputs.back(), layers.back());
  std::vector<DoubleVector> deltas = backpropagate_error(delta, weighted_inputs, activations);

  // Compare gradients
  double max_diff = 0.0;
  for (size_t l = 0; l < layers.size(); ++l) {
    // Compare bias gradients
    for (unsigned j = 0; j < layers[l].get_n_neurons(); ++j) {
      double diff = std::abs(deltas[l][j] - fd_bias_gradients[l][j]);
      max_diff = std::max(max_diff, diff);
    }

    // Compare weight gradients
    for (unsigned j = 0; j < layers[l].get_n_neurons(); ++j) {
      for (unsigned k = 0; k < layers[l].get_n_input(); ++k) {
        double bp_grad = deltas[l][j] * activations[l][k];
        double diff = std::abs(bp_grad - fd_weight_gradients[l](j, k));
        max_diff = std::max(max_diff, diff);
      }
    }
  }

  std::cout << "Maximum difference between backprop and finite diff gradients: "
            << max_diff << std::endl;
}

// Train the neural network using stochastic gradient descent
void NeuralNetwork::train(
    const std::vector<std::pair<DoubleVector, DoubleVector>> &training_data,
    const double &learning_rate,
    const double &tol_training,
    const unsigned &max_iter,
    const std::string &convergence_history_file_name) {

  if (!training_data.empty()) {
    std::cout << "Validating gradients..." << std::endl;
    validate_gradients(training_data[0].first, training_data[0].second);
  }

  // Initialize random number generator for shuffling
  std::default_random_engine generator(static_cast<unsigned>(std::time(0)));

  // Initialize iteration counter
  unsigned iter = 0;

  // Open convergence history file if provided
  std::ofstream convergence_file;
  if (!convergence_history_file_name.empty()) {
    convergence_file.open(convergence_history_file_name);
    if (!convergence_file.is_open()) {
      throw std::runtime_error("Unable to open convergence history file: " + convergence_history_file_name);
    }
  }

  // Training loop
  while (iter < max_iter) {
    // Shuffle the training data
    std::vector<std::pair<DoubleVector, DoubleVector>> shuffled_data = training_data;
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), generator);

    // Iterate over each training example
    for (const auto &data_pair : shuffled_data) {
      const DoubleVector &input = data_pair.first;
      const DoubleVector &target_output = data_pair.second;

      // Perform feed-forward to get the network's output and store intermediate results
      std::vector<DoubleVector> activations;
      std::vector<DoubleVector> weighted_inputs;
      feed_forward_with_intermediate_results(input, activations, weighted_inputs);

      // Compute the error at the output layer
      DoubleVector delta = compute_output_layer_delta(activations.back(), target_output, weighted_inputs.back(), layers.back());

      // Backpropagate the error and compute gradients
      std::vector<DoubleVector> deltas = backpropagate_error(delta, weighted_inputs, activations);

      // Update weights and biases
      update_weights_and_biases(deltas, activations, learning_rate);
    }

    // Compute the total cost for the training data
    double total_cost = cost_for_training_data(training_data);

    // Record convergence history if file is provided
    if (convergence_file.is_open()) {
      convergence_file << iter << " " << total_cost << std::endl;
    }

    // Check for convergence
    if (total_cost < tol_training) {
      break;
    }

    // Increment iteration counter
    ++iter;
  }

  // Close convergence history file if open
  if (convergence_file.is_open()) {
    convergence_file.close();
  }
}

std::mt19937 NeuralNetwork::random_generator(12345);

// Initialize weights and biases with random values
void NeuralNetwork::initialise_parameters(const double &mean, const double &std_dev) {
  // Create a random number generator and normal distribution
  std::normal_distribution<double> normal_dist(mean, std_dev);

  // Iterate over each layer in the network
  for (auto &layer : layers) {
    // Initialize biases
    for (unsigned i = 0; i < layer.biases.n(); ++i) {
      layer.biases[i] = normal_dist(random_generator);
    }

    // Initialize weights
    for (unsigned i = 0; i < layer.weights.n(); ++i) {
      for (unsigned j = 0; j < layer.weights.m(); ++j) {
        layer.weights(i, j) = normal_dist(random_generator);
      }
    }
  }
}

// Initialize weights and biases with a specific value for testing
void NeuralNetwork::initialise_parameters_for_test(const double &value) {
  // Iterate over each layer in the network
  for (auto &layer : layers) {
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