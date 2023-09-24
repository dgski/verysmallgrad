# Modern C++ Implementation of micrograd

A crude C++ implementation of @karpathy micrograd framework. Aka:

1. [engine.hpp](src/engine.hpp): Mathematical expression relation builder with backwards propagating gradient descent (partial derivatives).
2. [nn.hpp](src/nn.hpp): A simple neural net framework built using the engine.

Example use is demonstrated within [tests.cpp](src/tests.cpp)