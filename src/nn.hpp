#pragma once

#include "engine.hpp"
#include <random>

// N scalar inputs -> 1 scalar output
// Maintains a 'weight' multiplier for each input to manipulate effect
// Has an overall bias to control overall firing
class Neuron {
  std::vector<ValuePtr> _weights;
  ValuePtr _bias;

  static auto generateWeights(size_t numberOfInputs) {
    std::random_device rd{};
    std::mt19937 twister(rd());
    std::vector<ValuePtr> result;
    std::generate_n(std::back_inserter(result), numberOfInputs, [&]() {
      return Value::make(std::uniform_real_distribution<double>(-1.0, 1.0)(twister));
    });
    return result;
  }
public:
  Neuron(size_t numberOfInputs) : _weights(generateWeights(numberOfInputs)), _bias(Value::make(0.0))
  {}

  ValuePtr operator()(const std::vector<ValuePtr>& input) {
    auto sum = _bias;
    for (size_t i = 0; i<input.size(); ++i) {
      sum = sum + (input[i]* _weights[i]);
    }
    return sum;
  }

  auto parameters() {
    std::vector<Value*> params;
    for (auto& w : _weights) {
      params.push_back(w.get());
    }
    params.push_back(_bias.get());
    return params;
  }
};

// N scalar inputs -> X scalar outputs (X being number of neurons)
// Computed by feeding the inputs to each Neuron in the layer
// and including the single scalar output in the result
class Layer {
  std::vector<Neuron> _neurons;
public:
  Layer(size_t numberOfInputs, size_t numberOfOutputs) : _neurons(numberOfOutputs, Neuron(numberOfInputs))
  {}

  auto operator()(const std::vector<ValuePtr>& input) {
    std::vector<ValuePtr> result;
    for (auto& n : _neurons) {
      result.push_back(n(input));
    }
    return result;
  }

  auto parameters() {
    std::vector<Value*> params;
    for (auto& n : _neurons) {
      auto neuronParams = n.parameters();
      params.insert(params.end(), neuronParams.begin(), neuronParams.end());
    }
    return params;
  }
};

// N scalar inputs, M scalar outputs
// Feeds input through the next layer
// and the output to next consecutive layer
// until it reaches the end
class MultilayerPerceptron {
  std::vector<Layer> _layers;
public:
  MultilayerPerceptron(const std::vector<size_t>& neuronsPerLayer) {
    for (size_t i = 0; i<neuronsPerLayer.size()-1; ++i) {
      _layers.push_back(Layer(neuronsPerLayer[i], neuronsPerLayer[i+1]));
    }
  }

  auto operator()(std::vector<ValuePtr> input) {
    for (auto& l : _layers) {
      input = l(input);
    }
    return input;
  }

  auto parameters() {
    std::vector<Value*> params;
    for (auto& l : _layers) {
      auto layerParams = l.parameters();
      params.insert(params.end(), layerParams.begin(), layerParams.end());
    }
    return params;
  }
};