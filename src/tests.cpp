#include <cstdlib>
#include <iostream>
#include <cassert>

#include "engine.hpp"
#include "nn.hpp"

void engineTests()
{
  auto a = Value::make(2.0);
  auto b = Value::make(-3.0);
  auto c = Value::make(10.0);
  auto e = a * b;
  auto d = e + c;
  auto f = Value::make(2.0);
  auto L = d * f;
  auto lpow = power(L, -1);
  auto reluResult= relu(lpow);
  reluResult->_grad = 1.0;
  reluResult->backwards();

  assert(reluResult->_value == 0.125);
  assert(L->_value == 8.0);
  assert(L->_grad == -0.015625);
  assert(a->_grad == 0.09375);
}

auto print = [](auto& v) {
  std::cout << '[';
  for (auto& v : v) {
    std::cout << v->_value << ' ';
  }
  std::cout << ']' << std::endl;
};

auto checkLoss = [](auto& actual, auto& predicted) {
  auto result = Value::make(0.0);
  for (size_t i=0; i<actual.size(); ++i) {
    result = result + power((actual[i] - predicted[i]), 2.0);
  }
  return result;
};

void nnTests1()
{
  // prepare sample data
  auto xs = std::vector<std::vector<ValuePtr>>{
    { Value::make(0.0) },
    { Value::make(1.0) },
    { Value::make(0.0) },
    { Value::make(1.0) }
  };
  auto ys = std::vector<ValuePtr>{
    Value::make(1.0), Value::make(-1.0), Value::make(1.0), Value::make(-1.0)
  };

  // create and train neural net
  auto mlp = MultilayerPerceptron({ 1, 10, 10, 1 });
  for (size_t i=0; i<10000; ++i) {
    // Forwards: Run each training data through the neural net
    auto ypred = std::vector<ValuePtr>{};
    for (auto& x : xs) {
      ypred.push_back(mlp(x).front());
    }

    // Backwards: Compute loss and gradient descent 
    auto loss = checkLoss(ys, ypred);
    if (loss->_value < 0.0001) {
      break;
    }
    for (auto p : mlp.parameters()) {
        p->_grad = 0.0;
    }
    loss->backwards();

    // Update: Adjust weights and bias parameters
    for (auto p : mlp.parameters()) {
      p->_value -= (0.00001 * p->_grad);
    }
  }

  // See if training worked well
  auto input = std::vector<ValuePtr>{ Value::make(1.0) };
  auto pred = mlp(input);
  assert(pred.front()->_value < 0.0);

  auto input2 = std::vector<ValuePtr>{ Value::make(0.0) };
  auto pred2 = mlp(input2);
  assert(pred2.front()->_value > 0.0);
}

void nnTests2()
{
  // prepare sample data
  auto xs = std::vector<std::vector<ValuePtr>>{
    { Value::make(2.0), Value::make(3.0), Value::make(-1.0) },
    { Value::make(3.0), Value::make(-1.0), Value::make(0.5)},
    { Value::make(0.5), Value::make(1.0), Value::make(1.0) },
    { Value::make(1.0), Value::make(1.0), Value::make(-1.0) }
  };
  auto ys = std::vector<ValuePtr>{
    Value::make(1.0), Value::make(-1.0), Value::make(-1.0), Value::make(1.0)
  };

  // create and train neural net
  auto mlp = MultilayerPerceptron({ 3, 4, 4, 1 });
  for (size_t i=0; i<10000; ++i) {
    // Forwards: Run each training data through the neural net
    auto ypred = std::vector<ValuePtr>{};
    for (auto& x : xs) {
      ypred.push_back(mlp(x).front());
    }

    // Backwards: Compute loss and gradient descent 
    auto loss = checkLoss(ys, ypred);
    if (loss->_value < 0.00000000001) {
      break;
    }
    for (auto p : mlp.parameters()) {
      p->_grad = 0.0;
    }
    loss->backwards();

  // Update: Adjust weights and bias parameters
    for (auto p : mlp.parameters()) {
      p->_value -= (0.0001 * p->_grad);
    }
  }

  auto input = std::vector<ValuePtr>{ { Value::make(2.0), Value::make(3.0), Value::make(-1.0) } };
  auto pred = mlp(input);
  assert(std::abs(pred.front()->_value - 1.0) > 0.0);
}

int main() {
  engineTests();
  nnTests1();
  nnTests2();
  return EXIT_SUCCESS;
}