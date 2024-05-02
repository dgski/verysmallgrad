#pragma once

#include <vector>
#include <iostream>
#include <functional>

class Tensor {
  std::vector<double> _data;
  std::vector<size_t> _shape;
  std::vector<size_t> _strides;

  size_t getStride(size_t dim) const {
    return _strides[dim];
  }
  auto buildStrides() {
    std::vector<size_t> strides(_shape.size());
    strides.back() = 1;
    for (int i=_shape.size()-2; i>=0; --i) {
      strides[i] = strides[i+1] * _shape[i];
    }
    return strides;
  }
  static auto getSize(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (const auto& s : shape) {
      size *= s;
    }
    return size;
  }
public:
  Tensor(std::vector<double> data, std::vector<size_t> shape)
  : _data(data),
  _shape(shape),
  _strides(buildStrides())
  {
    if (getSize(shape) != data.size()) {
      throw std::runtime_error("Data size does not match shape");
    }
  }
  Tensor(double value)
  : _data({value}),
  _shape({1}),
  _strides({1})
  {}
  Tensor operator[](std::initializer_list<size_t> indices) const {
    size_t pos = 0;
    for (size_t i=0; i<indices.size(); ++i) {
      const auto stride = getStride(i);
      pos += *(indices.begin() + i) * stride;
    }

    if (indices.size() == _shape.size()) {
      return Tensor({_data[pos]}, {1});
    } else {
      std::vector<double> data;
      const auto shape = std::vector<size_t>(_shape.begin() + indices.size(), _shape.end());
      const auto size = getSize(shape);
      for (size_t i=0; i<size; ++i) {
        data.push_back(_data[pos + i]);
      }
      return Tensor(data, shape);
    }
  }
  double element() const {
    if (_data.size() != 1) {
      throw std::runtime_error("Tensor does not have a single element");
    }
    return _data[0];
  }
  Tensor view(std::initializer_list<size_t> shape) { return Tensor(_data, shape); }
  Tensor(const Tensor& other)
  : _data(other._data),
  _shape(other._shape),
  _strides(other._strides)
  {}
  auto& operator=(const Tensor& other) {
    _data = other._data;
    _shape = other._shape;
    _strides = other._strides;
    return *this;
  }
  auto& operator=(double value) {
    _data = {value};
    _shape = {1};
    _strides = {1};
    return *this;
  }

  template<typename Func>
  Tensor apply(Func&& fn) {
    std::vector<double> data;
    for (size_t i=0; i<_data.size(); ++i) {
      data.push_back(fn(_data[i], i));
    }
    return Tensor(data, _shape);
  }

  Tensor operator+(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d + other._data[i]; });
  }
  auto& operator+=(const Tensor& other) {
    _data = apply([&other](double d, size_t i) { return d + other._data[i]; })._data;
    return *this;
  }
  Tensor operator*(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d * other._data[i]; });
  }
  auto& operator*=(const Tensor& other) {
    _data = apply([&other](double d, size_t i) { return d * other._data[i]; })._data;
    return *this;
  }
  Tensor operator/(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d / other._data[i]; });
  }
  auto& operator/=(const Tensor& other) {
    _data = apply([&other](double d, size_t i) { return d / other._data[i]; })._data;
    return *this;
  }
  Tensor operator-(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d - other._data[i]; });
  }
  auto& operator-=(const Tensor& other) {
    _data = apply([&other](double d, size_t i) { return d - other._data[i]; })._data;
    return *this;
  }
  Tensor operator+(double value) {
    return apply([value](double d, size_t) { return d + value; });
  }
  auto& operator+=(double value) {
    _data = apply([value](double d, size_t) { return d + value; })._data;
    return *this;
  }
  Tensor operator*(double value) {
    return apply([value](double d, size_t) { return d * value; });
  }
  auto& operator*=(double value) {
    _data = apply([value](double d, size_t) { return d * value; })._data;
    return *this;
  }
  Tensor operator/(double value) {
    return apply([value](double d, size_t) { return d / value; });
  }
  auto& operator/=(double value) {
    _data = apply([value](double d, size_t) { return d / value; })._data;
    return *this;
  }
  Tensor operator-(double value) {
    return apply([value](double d, size_t) { return d - value; });
  }
  auto& operator-=(double value) {
    _data = apply([value](double d, size_t) { return d - value; })._data;
    return *this;
  }
  Tensor matmul(const Tensor& other) {
    // matrix multiplication is row by column
    // [a, b] * [c,
    //          d] = [a*c + b*d]
    // The resultant matrix will have the shape of [1st matrix rows, 2nd matrix columns]
    // The number of columns in the 1st matrix must be equal to the number of rows in the 2nd matrix
    assert(_shape.size() == 2);
    assert(other._shape == _shape);
    std::vector<double> data;
    for (size_t i=0; i<_shape[0]; ++i) {
      for (size_t j=0; j<other._shape[1]; ++j) {
        double sum = 0.0;
        for (size_t k=0; k<_shape[1]; ++k) {
          sum += _data[i*_shape[1] + k] * other._data[k*other._shape[1] + j];
        }
        data.push_back(sum);
      }
    }
    return Tensor(data, {_shape[0], other._shape[1]});
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    if (t._shape.size() > 2) {
      throw std::runtime_error("Only 1D and 2D tensors are supported");
    }
    if (t._shape.size() == 1) {
      os << '[';
      for (size_t i=0; i<t._shape[0]; ++i) {
        os << t._data[i] << ' ';
      }
      os << ']';
    } else {
      for (size_t i=0; i<t._shape[0]; ++i) {
        os << '[';
        for (size_t j=0; j<t._shape[1]; ++j) {
          os << t._data[i*t._shape[1] + j] << ' ';
        }
        os << ']' << std::endl;
      }
    }
    return os;
  }
  bool operator==(const Tensor& other) const {
    if (_shape != other._shape) {
      return false;
    }
    if (_data.size() != other._data.size()) {
      return false;
    }
    for (size_t i=0; i<_data.size(); ++i) {
      if (_data[i] != other._data[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const Tensor& other) const {
    return !(*this == other);
  }
  static Tensor fill(std::vector<size_t> shape, double value) {
    std::vector<double> data(getSize(shape), value);
    return Tensor(data, shape);
  }
  static Tensor zeros(std::vector<size_t> shape) { return fill(shape, 0.0); }
  static Tensor ones(std::vector<size_t> shape) { return fill(shape, 1.0); }
  static Tensor random(std::vector<size_t> shape) {
    std::vector<double> data;
    for (size_t i=0; i<getSize(shape); ++i) {
      data.push_back((double)rand() / RAND_MAX);
    }
    return Tensor(data, shape);
  }
  static Tensor single(double value) { return Tensor({value}, {1}); }
  const auto& shape() const { return _shape; }
  const auto& data() const { return _data; }
  auto sum() const {
    double result = 0.0;
    for (const auto& d : _data) {
      result += d;
    }
    return result;
  }
  auto relu() {
    return apply([](double d, size_t) { return d > 0.0 ? d : 0.0; });
  }
  auto power (double value) {
    return apply([value](double d, size_t) { return std::pow(d, value); });
  }

  bool operator<(const Tensor& other) const {
    return sum() < other.sum();
  }
  bool operator>(const Tensor& other) const {
    return sum() > other.sum();
  }
  bool operator<=(const Tensor& other) const {
    return sum() <= other.sum();
  }

  // Add single element comparison operators
  bool operator<(double value) const {
    assert(_data.size() == 1);
    return sum() < value;
  }
  bool operator>(double value) const {
    assert(_data.size() == 1);
    return sum() > value;
  }
  bool operator<=(double value) const {
    assert(_data.size() == 1);
    return sum() <= value;
  }
  bool operator==(double value) const {
    assert(_data.size() == 1);
    return sum() == value;
  }
  bool operator!=(double value) const {
    assert(_data.size() == 1);
    return sum() != value;
  }
};