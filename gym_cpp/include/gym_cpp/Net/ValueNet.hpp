#pragma once

#include <torch/torch.h>

#include <trpo/model/ValueNetBase.hpp>

struct ValueNet : ValueNetBase {
  ValueNet(int64_t N,int64_t H):
    linear1(register_module("linear1", torch::nn::Linear(N  , H) ) ),
    linear2(register_module("linear2", torch::nn::Linear(H,   H) ) ),
    linear3(register_module("linear3", torch::nn::Linear(H,   H) ) ),
    linear4(register_module("linear4", torch::nn::Linear(H,   1) ) )
  {
  }

  torch::Tensor forward(torch::Tensor input) override {
    return  linear4(
            torch::tanh( linear3(
            torch::tanh( linear2(
            torch::tanh( linear1(input
            ) )
            ) )
            ) )
            ) ;
  }
  torch::nn::Linear linear1;
  torch::nn::Linear linear2;
  torch::nn::Linear linear3;
  torch::nn::Linear linear4;

};
