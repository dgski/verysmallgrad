#pragma once

#include "engine.hpp"
#include <random>
#include <list>

/*
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
*/

class Neuron {
  std::list<Value> _weights;
  Value _bias;

  static auto generateWeights(size_t count) {
    std::random_device rd{};
    std::mt19937 twister(rd());
    std::list<Value> result;
    std::generate_n(std::back_inserter(result), count, [&]() {
      return Value(std::uniform_real_distribution<double>(-1.0, 1.0)(twister));
    });
    return result;
  }
public:
  Neuron(size_t count) : _weights(generateWeights(count)), _bias(0.0)
  {}

  auto operator()() {
    //TODO: implement
  }

  auto parameters() {
    std::vector<Value*> params;
    for (auto& w : _weights) {
      params.push_back(&w);
    }
    params.push_back(&_bias);
    return params;
  }
};