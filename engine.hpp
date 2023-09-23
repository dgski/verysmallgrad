#pragma once

#include <array>
#include <ostream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

enum class Operation {
  Null,
  Addition,
  Multiplication
};

std::string_view toString(Operation op)
{
  switch (op)
  {
    case Operation::Null: return "null";
    case Operation::Addition: return "+";
    case Operation::Multiplication: return "*";
  }

  throw std::runtime_error("Unhandled op");
}

struct Value;

struct Inputs {
  Operation operation = Operation::Null;
  std::vector<Value*> values;
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
      buildTopo(topo, visited, next);
    }
    topo.push_back(value);
  }
public:
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
      a->_grad += b->_value;
      b->_grad += a->_value;
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

  Value operator+(Value& other)
  {
    return Value(_value + other._value, Inputs{ Operation::Addition, { this, &other }});
  }
  Value operator*(Value& other)
  {
    return Value(_value * other._value, Inputs{ Operation::Multiplication, { this, &other }});
  }
  friend std::ostream& operator<<(std::ostream& os, const Value& value)
  {
    os << value._value;
    return os;
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
};

