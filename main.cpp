#include <cstdlib>
#include <iostream>

#include "engine.hpp"

int main() {
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
    reluResult->printTree();


    return EXIT_SUCCESS;
}