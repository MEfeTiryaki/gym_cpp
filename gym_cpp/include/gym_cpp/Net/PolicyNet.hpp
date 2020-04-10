#pragma once

#include <torch/torch.h>
#include <trpo/model/StochasticPolicyNetBase.hpp>


struct PolicyNet :StochasticPolicyNetBase {
  PolicyNet(int64_t N,int64_t M,int64_t H):
    StochasticPolicyNetBase(),
    inputLayer(register_module("inputLayer", torch::nn::Linear(N  , H) ) ),
    hiddenLayer(register_module("hiddenLayer", torch::nn::Linear(H, H) ) ),
    hiddenLayer2(register_module("hiddenLayer2", torch::nn::Linear(H, H) ) ),
    outputLayer(register_module("outputLayer", torch::nn::Linear(H,   M) ) ),
    outputSize_(M),
    inputSize_(N),
    hiddenLayerSize_(H)
  {
      register_parameter("logStd",torch::zeros({1,M},at::kDouble) );
  }

  torch::Tensor forward(torch::Tensor input) override {
    actionMeans_ =  outputLayer(
                    torch::tanh( hiddenLayer2(
                    torch::tanh( hiddenLayer(
                    torch::tanh( inputLayer(input
                    ) )
                    ) )
                    ) )
                    ) ;

    logStd_ = named_parameters()["logStd"].expand_as(actionMeans_);
    std_= torch::exp(logStd_);
    actions_ = torch::zeros(actionMeans_.sizes());
    return actionMeans_ ;
  }

  void setLogStd(double x){
    named_parameters()["logStd"] = x*torch::ones({1,outputSize_},at::kDouble);
  }
  torch::Tensor getLogProbabilityDensity(torch::Tensor states
                                        ,torch::Tensor actions) override {

    torch::Tensor actionMeans = forward(states);
    torch::Tensor var = torch::exp(logStd_).pow(2);
    logProbablitiesDensity_ = -(actions - actionMeans).pow(2) / (
            2 * var) - 0.5 * log(2 * M_PI) - logStd_;
    return logProbablitiesDensity_.sum(1);
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
    actions_ = torch::zeros({robotNumber,outputSize_});
    actions_ = at::normal(actionMeans,std_);
  }

  virtual void getActions(double* actions, int robotNumber) override{
    double* actionPointer = actions_.data_ptr<double>();
    for(int i = 0 ; i< robotNumber ;i++){
      for(int j = 0 ; j< outputSize_ ;j++){
      *(actions+ outputSize_*i +j)  =  *(actionPointer+ outputSize_*i +j);
      }
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
