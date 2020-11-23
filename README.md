# PySnake

An AI-enabled Snake that allows you to train a PyTorch neural network and see the trained network play the game in a Tkinter application. 
This project was made by Brady Simon and Gerald Arenas as a way to learn more about neural networks and how representing the same 
information differently can cause different outcomes.

There are three different neural network files:
  1. NeuralNetwork.py
  2. VisualNeuralNetwork.py
  3. GeneticSnakeAI.py

Once a model is trained (or loaded from one of the stored versions), you can set it as a snake's controller and have it play the game 
through a _SnakeWindow_ instance.

## General Information
A _Board_ object stores all of the marks on the board. The _SnakeBoard_ manages the _Snake_ instances on the board and updates the board.
Snakes have a _SnakeControllable_ instance called _controller_ that defines which direction a snake should move given a _SnakeBoard_ and the 
snake's name. 

_SnakeWindow_ can play the given _SnakeBoard_ until the game ends, at which point a reset function can continuously play the game.
Each neural network implements _SnakeControllable_ so that _SnakeBoard_ can automatically update its desired direction when requested.

## SnakeWindow.py
This is a Tkinter window that plays the given _SnakeBoard_ instance. It allows either AI-controlled snakes or human-controlled snakes, 
as well as other customizations like gradients, toggling gridlines, and a snake health bar. 

## NeuralNetwork.py
A simple starter neural network that attempts to train a snake neural network based off recorded games played by people. 
This model only has a few inputs to keep the model simple:
  1. Safe directions (up, down, left, right around snake's head, binary)
  2. Length of snake (segments / 100)
  2. Directions to fruit (estimated by a number between 0 and 1)
  
Total Input: 5-length Tensor.

Total Output: 4-length Tensor (direction to choose).

## VisualNeuralNetwork.py
A larger version of _NeuralNetwork_ that uses more inputs in an attempt to achieve a higher score. 
This model has several inputs, but trains the same way as _NeuralNetwork_:
  1. Current snake direction (up, down, left, right, 0/1)
  2. Directions to point (up, down, left, right, up/right, down/left, etc, binary)
  3. Safe directions (up, down, left, right around snake's head, binary)
  4. Cardinal vision around the snake (8 directions, 3 values each: fruit in sight, snake in sight, distance to wall)
  
Total Input: 36-length Tensor.

Total Output: 4-length Tensor (direction to choose).
 
## GeneticSnakeAI.py
Functions identically to _VisualNeuralNetwork_ but instead is trained through a genetic algorithm. 
The _GeneticTrainer_ class is given a PyTorch state dictionary and its accompanying model and parameters.
The _GeneticTrainer_ creates the initial population by randomizing the original state dictionary and then 
simulates each snake in the population. The snakes with a higher than average fitness live onto the next 
generation and crossover their genes with each other, filling in the remaining population positions. 

This cycle continues until one of the exit conditions are met:
  - The number of generations have been met.
  - A desired cutoff fitness has been reached.
  - Training time has exceeded a given timeout.
  
This training process takes considerably longer than the previous networks, but yields greater results due to overcoming 
human limitations in training data. Further improvements could build upon the inputs to the network (either by adding more
or reworking the vision system), but a general solution was desired as opposed to passing in the entire _SnakeBoard_ instance
and restricting the model to a certain board layout.
