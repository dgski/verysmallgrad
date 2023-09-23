#include <cstdlib>
#include <iostream>

#include "engine.hpp"
#include "nn.hpp"

int main() {
    auto a = Value(2.0);
    auto b = Value(-3.0);
    auto c = Value(10.0);
    auto e = a * b;
    auto d = e + c;
    auto f = Value(2.0);
    auto L = d * f;

    L._grad  = 1.0;
    L.backwards();
    L.printTree();

    return EXIT_SUCCESS;
}