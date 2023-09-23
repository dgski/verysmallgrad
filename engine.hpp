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

enum class Operation {
  Null,
  Addition,
  Multiplication,
  Power,
  RELU
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
  }

  throw std::runtime_error("Unhandled op");
}

struct Value;

struct Inputs {
  Operation operation = Operation::Null;
  std::vector<std::shared_ptr<Value>> values;
  double power = 0.0;
};

struct Value {
  double _value;
  Inputs _inputs;
  double _grad = 0.0;

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
  : _value(value), _inputs(inputs)
  {}

  void zeroGrad() {
    _grad = 0.0;
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
      _inputs.values[0]->_grad += (_inputs.power * std::pow(_value, _inputs.power-1)) * _grad;
    } else if (_inputs.operation == Operation::RELU) {
      _inputs.values[0]->_grad = double(_value > 0.0) * _grad;
    }
  }

  void backwards()
  {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    buildTopo(topo, visited, this);
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
  return std::make_shared<Value>(std::pow(a->_value, value), Inputs{ Operation::Power, { a }, value });
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
  return std::make_shared<Value>((a->_value > 0.0 ? a->_value : 0.0), Inputs{ Operation::RELU, { a } });
}

std::ostream& operator<<(std::ostream& os, const ValuePtr& value)
{
  os << value->_value;
  return os;
}