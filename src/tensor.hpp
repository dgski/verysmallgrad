#pragma once

#include <vector>

class Tensor {
  std::vector<double> _data;
  std::vector<size_t> _shape;

  size_t getStride(size_t dim) const {
    size_t stride = 1;
    for (size_t i=dim+1; i<_shape.size(); ++i) {
      stride *= _shape[i];
    }
    return stride;
  }
public:
  Tensor(std::vector<double> data, std::vector<size_t> shape) : _data(data), _shape(shape) {}
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
      for (size_t i=0; i<_shape[indices.size()]; ++i) {
        data.push_back(_data[pos + i]);
      }
      return Tensor(data, {_shape[indices.size()]});
    }
  }
  double element() const {
    if (_data.size() != 1) {
      throw std::runtime_error("Tensor does not have a single element");
    }
    return _data[0];
  }
  Tensor view(std::initializer_list<size_t> shape) { return Tensor(_data, shape); }
  Tensor operator+(const Tensor& other) {
    std::vector<double> result;
    for (size_t i=0; i<_data.size(); ++i) {
      result.push_back(_data[i] + other._data[i]);
    }
    return Tensor(result, _shape);
  }
  // Implement element-wise multiplication
  Tensor operator*(const Tensor& other) {
    std::vector<double> result;
    for (size_t i=0; i<_data.size(); ++i) {
      result.push_back(_data[i] * other._data[i]);
    }
    return Tensor(result, _shape);
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

  static Tensor zeros(std::vector<size_t> shape) {
    std::vector<double> data;
    for (size_t i=0; i<shape[0]; ++i) {
      data.push_back(0.0);
    }
    return Tensor(data, shape);
  }
  static Tensor ones(std::vector<size_t> shape) {
    std::vector<double> data;
    for (size_t i=0; i<shape[0]; ++i) {
      data.push_back(1.0);
    }
    return Tensor(data, shape);
  }
};