# Neural Network

06 June 2011

A neural network implementation made for the *Machine Learning* course at the University of Pisa.

## Description

Allows you to create and test a neural network for classification with the following characteristics:

  - The outputs are in the range (0,1)
  - Each layers is fully connected with the follow layer and only to that
  - The activation function of each unit is the sigmoid function f(x) = 1/(1+e^(-x))
  - The weights of each unit are initialized randomly in the range [-0.7,+0.7] except the 0.
  - Allows an arbitrary number of inputs, outputs and hidden layers with an arbitrary number of units

Depending on the mode selected (with parameter `--mode`) you can:

  - Create a custom neural network and train it with the back-propagation algorithm on a dataset passed in csv format (mode training)
  - Test a neural network, previously created, on a dataset in csv format (mode test)

In general, the format of the file containing the dataset must be:

    id, input[1], ..., input[n], output[1], ..., output[m]

where

  - each row is an instance with a unique identifier id
  - one instance has n inputs and m outputs
  - each instance has the same number of inputs and outputs to the others

In test mode you can have a dataset without outputs to get the answers by the neural network.

See `help.txt` for more.
