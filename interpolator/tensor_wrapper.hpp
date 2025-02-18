#pragma once
#include <torch/torch.h>

#ifndef PRECISION
#define PRECISION double
#endif

class TensorWrapper {
public:
    explicit TensorWrapper(const torch::Tensor& tensor) : tensor_(tensor) {}
    
    PRECISION* ptr() const {
        return tensor_.data_ptr<PRECISION>();
    }
private:
    torch::Tensor tensor_;
}; 