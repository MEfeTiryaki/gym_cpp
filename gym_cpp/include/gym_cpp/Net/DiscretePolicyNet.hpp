#pragma once

#include <torch/torch.h>
#include <trpo/model/StochasticPolicyNetBase.hpp>


struct DiscretePolicyNet :StochasticPolicyNetBase {
  DiscretePolicyNet(int64_t N,int64_t M,int64_t H):
    StochasticPolicyNetBase(),
    inputLayer(register_module("inputLayer", torch::nn::Linear(N  , H) ) ),
    hiddenLayer(register_module("hiddenLayer", torch::nn::Linear(H, H) ) ),
    hiddenLayer2(register_module("hiddenLayer2", torch::nn::Linear(H, H) ) ),
    outputLayer(register_module("outputLayer", torch::nn::Linear(H,   M) ) ),
    outputSize_(M),
    inputSize_(N),
    hiddenLayerSize_(H)
  {
  }

  torch::Tensor forward(torch::Tensor input) override {
    actionMeans_ =  torch::softmax( outputLayer(
                    torch::tanh( hiddenLayer2(
                    torch::tanh( hiddenLayer(
                    torch::tanh( inputLayer(input
                    ) )
                    ) )
                    ) )
                  ) ,1);

    actions_ = torch::zeros({actionMeans_.sizes()[0],1});
    return actionMeans_ ;
  }


  torch::Tensor getLogProbabilityDensity(torch::Tensor states
                                        ,torch::Tensor actions) override {

    torch::Tensor actionMeans = forward(states);
    torch::Tensor prob = torch::zeros(actions.sizes()[0]);
    for(int i =0 ;i < actions.sizes()[0]; i++){
      prob[i] = actionMeans[i][(int)actions[i].item<double>()];
    }
    return log(prob);
  }

  torch::Tensor meanKlDivergence(torch::Tensor states,
                                 torch::Tensor actions,
                                 torch::Tensor logProbablityOld) override{
    // Get states
    torch::Tensor logProbabilityNew = getLogProbabilityDensity(
                                      states
                                     ,actions);
    return (torch::exp(logProbablityOld)
            * (logProbablityOld - logProbabilityNew)).mean(); //Tensor kl.mean()
  }

  void calculateAction(double* state , int robotNumber, int stateSize) override{
    // zero_grad();
    torch::Tensor stateTensor =  torch::zeros({robotNumber,stateSize},at::kDouble);
    double* stateTensorPointer = stateTensor.data_ptr<double>() ;

    for(int i = 0 ; i< robotNumber ;i++){
      for(int j = 0 ; j< stateSize ;j++){
        *(stateTensorPointer+stateSize*i+j)= *(state + stateSize*i +j) ;//Vf.data();
      }
    }
    torch::Tensor actionMeans = forward(stateTensor);


    actions_ = torch::zeros({actionMeans.sizes()[0],1},at::kDouble);
    actions_ = at::multinomial(actionMeans,1).to(torch::kDouble);;
  }

  virtual void getActions(double* actions, int robotNumber) override{
    double* actionPointer = actions_.data_ptr<double>();
    for(int i = 0 ; i< robotNumber ;i++){
      *(actions+ outputSize_*i)  =  *(actionPointer+ outputSize_*i);
    }
  }

  virtual void getMeanActions(double* actions, int robotNumber) override{

    double* actionMeansPointer = actionMeans_.data_ptr<double>();
    for(int i = 0 ; i< robotNumber ;i++){
      for(int j = 0 ; j< outputSize_ ;j++){
      *(actions+ outputSize_*i +j)  = *(actionMeansPointer+ outputSize_*i +j);//Vf.data();
      }
    }
  }

  torch::nn::Linear inputLayer;
  torch::nn::Linear hiddenLayer;
  torch::nn::Linear hiddenLayer2;
  torch::nn::Linear outputLayer;


  int inputSize_;
  int outputSize_;
  int hiddenLayerSize_;

  torch::Tensor std_;
};
