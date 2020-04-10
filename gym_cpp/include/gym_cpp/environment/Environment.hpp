#pragma once

#include "opencv2/opencv.hpp"

namespace gym{
  class Environment{
    public:
      Environment(){
      }

      ~Environment(){}

      virtual bool step(Eigen::VectorXd& newState, double& reward, Eigen::VectorXd u){
      }

      virtual Eigen::VectorXd reset(){
      }

      virtual void close(){
      }

      virtual cv::Mat renderVideo(){
        return cv::Mat();
      }

      virtual void render(){
      }

      virtual int getStateSize(){
        return  0;
      }

      virtual int getActionSize(){
        return  0;
      }
  };
}// namespace gym
