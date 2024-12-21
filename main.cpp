#include "project2_a.h"

#include <chrono>

int main()
{

  std::string training_data_file = "spiral_training_data.dat";
  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
  ActivationFunction *activation_function_pt = new TanhActivationFunction;

  // Build the network: 2, 16, 16, 1 neurons in the layers
  // Number of neurons in the input layer
  unsigned n_input = 2;

  // Storage for the non-input layers: number of neurons and activation function
  std::vector<std::pair<unsigned, ActivationFunction *>> non_input_layer;

  // First hidden layer with 16 neurons
  non_input_layer.push_back(std::make_pair(16, activation_function_pt));

  // Second hidden layer with 16 neurons
  non_input_layer.push_back(std::make_pair(16, activation_function_pt));

  // Output layer with 1 neuron
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));

  // Create the NeuralNetwork instance
  NeuralNetwork network(n_input, non_input_layer);
  network.initialise_parameters(0.0, 0.1);
  network.read_training_data(training_data_file, training_data);

  // Train the network
  double learning_rate = 0.01;
  double tol_training = 0.001;
  unsigned max_iter = 20000;
  std::string convergence_history_file = "convergence_history (2, 16, 16, 1)_stopped.txt";

  auto start_time = std::chrono::high_resolution_clock::now();
  network.train(training_data, learning_rate, tol_training, max_iter, convergence_history_file);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end_time - start_time;

  // Output results to a file
  std::ofstream outfile("solve_times (2, 16, 16, 1)_stopped.txt");
  outfile << "training time\n";
  outfile << time.count();
  outfile.close();

  // Evaluate the network on the training data
  double total_cost = network.cost_for_training_data(training_data);
  std::cout << "Total cost for training data after training: " << total_cost << std::endl;

  // Save the trained parameters to a file
  std::ofstream boundary_file("boundary_data (2, 16, 16, 1)_stopped.txt");
  for (double x1 = 0.0; x1 <= 1.0; x1 += 0.01)
  {
    for (double x2 = 0.0; x2 <= 1.0; x2 += 0.01)
    {
      DoubleVector input(2);
      input[0] = x1;
      input[1] = x2;
      DoubleVector output;
      network.feed_forward(input, output);
      boundary_file << x1 << " " << x2 << " " << output[0] << std::endl;
    }
  }
  boundary_file.close();

  // Clean up
  delete activation_function_pt;

  return 0;
}