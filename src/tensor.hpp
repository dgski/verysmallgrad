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
  Tensor operator*(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d * other._data[i]; });
  }
  Tensor operator/(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d / other._data[i]; });
  }
  Tensor operator-(const Tensor& other) {
    return apply([&other](double d, size_t i) { return d - other._data[i]; });
  }
  Tensor operator+(double value) {
    return apply([value](double d, size_t) { return d + value; });
  }
  Tensor operator*(double value) {
    return apply([value](double d, size_t) { return d * value; });
  }
  Tensor operator/(double value) {
    return apply([value](double d, size_t) { return d / value; });
  }
  Tensor operator-(double value) {
    return apply([value](double d, size_t) { return d - value; });
  }
  Tensor matmul(const Tensor&) {
    // Implement matrix multiplication for N-dimensional tensors
    throw std::runtime_error("Not implemented");
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
  static Tensor fill(std::vector<size_t> shape, double value) {
    std::vector<double> data(getSize(shape), value);
    return Tensor(data, shape);
  }
  static Tensor zeros(std::vector<size_t> shape) { return fill(shape, 0.0); }
  static Tensor ones(std::vector<size_t> shape) { return fill(shape, 1.0); }
  const auto& shape() const { return _shape; }
  const auto& data() const { return _data; }
  auto sum() const {
    double result = 0.0;
    for (const auto& d : _data) {
      result += d;
    }
    return result;
  }
};