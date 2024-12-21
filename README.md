# Neural Network Binary Classification Project

This project implements a neural network from scratch in C++ to solve binary classification tasks. It includes object-oriented design principles, custom activation functions, and visualization of decision boundaries.

## Project Overview

This project builds a feed-forward fully connected neural network to classify data points in a 2D space. The training is performed using stochastic gradient descent with back-propagation, while results are visualized as decision boundaries.

## Features
- Custom neural network implementation with flexible architecture
- Object-oriented design for modularity
- Includes tanh activation function and extensible framework for others
- Training via stochastic gradient descent
- Visualization scripts to display decision boundaries

## Prerequisites
- C++11 or later
- Compiler (e.g., g++)
- Python with matplotlib (for visualization)

## Code Structure
- main.cpp: Driver code to initialize, train, and evaluate the network.
- project2_a.h: Main implementation of the neural network, layers, and training logic.
- project2_a_basics.h: Interfaces for activation functions and utility functions. File was given in the task
- dense_linear_algebra.h: Custom implementation of linear algebra operations, including matrices and vectors. File was given in the task
- plot.ipynb: Python notebook for visualizing decision boundaries and cost evolution.

Key Classes and Functions
- NeuralNetwork:
  - train(): Trains the network using stochastic gradient descent.
  - feed_forward(): Computes the network's output.
  - cost(): Calculates the cost for a single input-output pair.
- NeuralNetworkLayer:
  - Handles per-layer computation of activations and gradients.
- TanhActivationFunction:
 - Provides the tanh function and its derivative.

## Results
- Achieved binary classification for the spiral dataset.
- Visualized decision boundaries and convergence history.
- Demonstrated modularity for testing different architectures and activation functions.

References

1. Higham, C.F., & Higham, D.J. (2019). Deep Learning: An Introduction for Applied Mathematicians. SIAM Review, 61, 860-891.

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://www.linkedin.com/in/nikolay-kraynev-011337231/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="jksbjknfkjd" height="30" width="40" /></a>
<a href="https://www.facebook.com/profile.php?id=61553677926228" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg" alt="sdfghjkl" height="30" width="40" /></a>
<a href="https://t.me/king_nikoly" target="blank"><img align="center" src="https://img.icons8.com/?size=100&id=oWiuH0jFiU0R&format=png&color=000000" alt="sdfghjkl" height="30" width="40" /></a>
</p>
</p>
