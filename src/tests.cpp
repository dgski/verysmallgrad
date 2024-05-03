#include <cstdlib>
#include <iostream>
#include <cassert>

#include "engine.hpp"
#include "nn.hpp"
#include "tensor.hpp"

void tensorTests()
{
  auto ones = Tensor::ones({ 2, 2 });
  assert((ones[{0, 0}].element() == 1.0));
  assert((ones[{0, 1}].element() == 1.0));

  auto zeros = Tensor::zeros({ 2, 2 });
  assert((zeros[{0, 0}].element() == 0.0));
  assert((zeros[{0, 1}].element() == 0.0));

  auto filled = Tensor::fill({ 2, 2 }, 5.0);
  assert((filled[{0, 0}].element() == 5.0));
  assert((filled[{0, 1}].element() == 5.0));

  Tensor t({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 });
  assert((t[{1, 1}].element() == 4.0));
  assert((t[{ 1 }] == Tensor({ 3.0, 4.0 }, { 2 })));

  Tensor t1(
    { 1.0, 2.0,
    3.0, 4.0 }, { 2, 2 });
  Tensor t2(
    { 5.0, 6.0,
    7.0, 8.0 }, { 2, 2 });

  // (1*5 + 2*7), (1*6 + 2*8)
  // (3*5 + 4*7), (3*6 + 4*8)
  // = 19, 22
  // = 43, 50
  auto t3 = t1 + t2;
  assert((t3[{0, 0}].element() == 6.0));
  assert((std::stringstream() << t3).str() == "[[6 8 ][10 12 ]]");

  auto t4 = t1 * t2;
  assert((t4[{0, 0}].element() == 5.0));

  auto t5 = t1.matmul(t2);
  assert((t5 == Tensor({ 19.0, 22.0,
                        43.0, 50.0 }, { 2, 2 })));

  // Big matrix multiplication
  auto t6 = Tensor::ones({ 1000, 1000 });
  auto t7 = Tensor::ones({ 1000, 1000 });
  auto t8 = t6.matmul(t7);
  assert((t8[{0, 0}].element() == 1000.0));

  auto t9 = Tensor::ones({ 2, 2 });
  auto t10 = t9.relu();
  assert((t10[{0, 0}].element() == 1.0));

  auto t11 = Tensor::ones({ 2, 2 });
  auto t12 = t11.sum();
  assert((t12 == 4.0));

  auto t13 = Tensor::fill({ 2, 2 }, 2.0);
  auto t14 = t13.power(2.0);
  assert((t14[{0, 0}].element() == 4.0));

  auto t15 = Tensor({ 1.0 }, { 1 });
  assert((t15 < 2.0));

  // Transpose
  auto t16 = Tensor({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 });
  auto t17 = t16.transpose();
  assert((t17 == Tensor({ 1.0, 3.0, 2.0, 4.0 }, { 2, 2 })));
}

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

void engineTensorTests()
{
  auto a = Value::make(Tensor({ 1.0, 2.0, 3.0, 4.0 }, { 2, 2 }));
  auto b = Value::make(Tensor({ 5.0, 6.0, 7.0, 8.0 }, { 2, 2 }));
  auto c = a * b;

  assert(c->_value == Tensor({ 5.0, 12.0, 21.0, 32.0 }, { 2, 2 }));
  c->backwards();

  auto d = matmul(a, b);
  assert(d->_value == Tensor({ 19.0, 22.0, 43.0, 50.0 }, { 2, 2 }));
  d->zeroAllGrads();
  d->backwards();
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
  auto params = mlp.parameters();
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
    for (auto p : params) {
      p->_grad = 0.0;
    }
    loss->backwards();

    // Update: Adjust weights and bias parameters
    for (auto p : params) {
      p->_value -= (p->_grad * 0.00001);
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
  auto params = mlp.parameters();
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
    for (auto p : params) {
      p->_grad = 0.0;
    }
    loss->backwards();

  // Update: Adjust weights and bias parameters
    for (auto p : params) {
      p->_value -= (p->_grad * 0.0001);
    }
  }

  auto input = std::vector<ValuePtr>{ { Value::make(2.0), Value::make(3.0), Value::make(-1.0) } };
  auto pred = mlp(input);
  assert(std::abs(pred.front()->_value.element() - 1.0) > 0.0);
}

int main() {
  tensorTests();
  engineTests();
  engineTensorTests();

  nnTests1();
  nnTests2();
  return EXIT_SUCCESS;
}