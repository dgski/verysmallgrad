#pragma once

#include <array>
#include <ostream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <memory>

#include "tensor.hpp"

enum class Operation {
  Null,
  Addition,
  Multiplication,
  Power,
  RELU,
  MatMul,
  Sum
};

std::string_view toString(Operation op)
{
  switch (op)
  {
    case Operation::Null: return "null";
    case Operation::Addition: return "+";
    case Operation::Multiplication: return "*";
    case Operation::Power: return "pow";
    case Operation::RELU: return "RELU";
    case Operation::MatMul: return "MatMul";
    case Operation::Sum: return "Sum";
  }

  throw std::runtime_error("Unhandled op");
}

struct Value;

struct Inputs {
  Operation operation = Operation::Null;
  std::vector<std::shared_ptr<Value>> values;
  double power = 0.0;
};

// Note: do not construct directly unless you have specific requirements
// Use the Value::make(...), as the ValuePtr type has all the operators defined on it
//
// Scalar floating point number type which allows building and evaluating
// mathematical expression trees forwards and backward:
// - Forwards: resolve/simplify the mathematical expression value
// - Backwards: calculate the partial derivative for all input terms in the tree
// by applying the chain rule backwards
//
// This is done by saving the input expressions/terms for each 'Value'
// and traversing the tree as needed.
struct Value {
  Tensor _value;
  Inputs _inputs;
  Tensor _grad = 0.0;

  static void buildTopo(
    std::vector<Value*>& topo,
    std::unordered_set<Value*>& visited,
    Value* value)
  {
    if (visited.find(value) != visited.end()) {
      return;
    }

    visited.insert(value);
    for (auto next : value->_inputs.values) {
      buildTopo(topo, visited, next.get());
    }
    topo.push_back(value);
  }

  Value(double value, Inputs inputs = Inputs{})
  : _value(Tensor::single(value)), _inputs(inputs)
  {}
  Value(Tensor value, Inputs inputs = Inputs{})
  : _value(value), _inputs(inputs), _grad(Tensor::zeros(value.shape()))
  {}

  void zeroGrad() {
    _grad = _grad.apply([](double, size_t) { return 0.0; });
  }
  void zeroAllGrads() {
    zeroGrad();
    for (auto& value : _inputs.values) {
      value->zeroAllGrads();
    }
  }

  void backwardsOnce() {
    if (_inputs.operation == Operation::Addition) {
      _inputs.values[0]->_grad += _grad;
      _inputs.values[1]->_grad += _grad;
    } else if (_inputs.operation == Operation::Multiplication) {
      auto& a = _inputs.values[0];
      auto& b = _inputs.values[1];
      a->_grad += b->_value * _grad;
      b->_grad += a->_value * _grad;
    } else if (_inputs.operation == Operation::Power) {
      _inputs.values[0]->_grad += (_inputs.values[0]->_value.power(_inputs.power-1) * _inputs.power) * _grad;
    } else if (_inputs.operation == Operation::RELU) {
      _inputs.values[0]->_grad += _grad * _value.apply([](double x, size_t) { return x > 0.0; });
    } else if (_inputs.operation == Operation::MatMul) {
      auto& a = _inputs.values[0];
      auto& b = _inputs.values[1];
      // 4x1
      // 4x4
      // 4x4
      a->_grad += _grad.matmul(b->_value.transpose());
      b->_grad += a->_value.transpose().matmul(_grad);
    } else if (_inputs.operation == Operation::Sum) {
      _inputs.values[0]->_grad += _grad.element();
    }
  }

  void backwards()
  {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    buildTopo(topo, visited, this);
    _grad = _grad.apply([](double, size_t) { return 1.0; });
    std::for_each(std::rbegin(topo), std::rend(topo), [&](Value* value) {
      value->backwardsOnce();
    });
  }

  void printTree(int indents = 0)
  {
    std::string_view operation = _inputs.operation != Operation::Null ? toString(_inputs.operation)  : "";
    std::stringstream current;
    current << std::string(indents, ' ') << "value=" << _value << " grad=" << _grad << " " << operation << std::endl;
    const auto currentStr = current.str();

    if (const bool hasLeft = _inputs.values.size() > 0; hasLeft) {
      _inputs.values[0]->printTree(currentStr.size());
    }
    std::cout << currentStr;
    if (const bool hasRight = _inputs.values.size() > 1; hasRight) {
      _inputs.values[1]->printTree(currentStr.size());
    }
  }

  template<typename... Args>
  static auto make(Args&&... args)
  {
    return std::make_shared<Value>(std::forward<Args>(args)...);
  }

  auto params()
  {
    std::vector<Value*> params;
    std::unordered_set<Value*> visited;
    buildTopo(params, visited, this);
    return params;
  }
};

using ValuePtr = std::shared_ptr<Value>;

ValuePtr operator+(ValuePtr a, ValuePtr b)
{
  return std::make_shared<Value>(a->_value + b->_value, Inputs{ Operation::Addition, { a, b }});
}
ValuePtr operator*(ValuePtr a, ValuePtr b)
{
  return std::make_shared<Value>(a->_value * b->_value, Inputs{ Operation::Multiplication, { a, b }});
}
ValuePtr power(ValuePtr a, double value)
{
  return std::make_shared<Value>(a->_value.power(value), Inputs{ Operation::Power, { a }, value });
}
ValuePtr operator-(ValuePtr a)
{
  return a * std::make_shared<Value>(-1.0);
}
ValuePtr operator/(ValuePtr a, ValuePtr b)
{
  return a * power(b, -1.0);
}
ValuePtr operator-(ValuePtr a, ValuePtr b)
{
  return a + (-b);
}
ValuePtr relu(ValuePtr a)
{
  auto value = a->_value.apply([](double x, size_t) { return x > 0.0 ? x : 0.0; });
  return std::make_shared<Value>(std::move(value), Inputs{ Operation::RELU, { a } });
}
ValuePtr matmul(ValuePtr a, ValuePtr b)
{
  return std::make_shared<Value>(a->_value.matmul(b->_value), Inputs{ Operation::MatMul, { a, b } });
}
ValuePtr sum(ValuePtr a)
{
  return std::make_shared<Value>(a->_value.sum(), Inputs{ Operation::Sum, { a } });
}

std::ostream& operator<<(std::ostream& os, const ValuePtr& value)
{
  os << value->_value;
  return os;
}