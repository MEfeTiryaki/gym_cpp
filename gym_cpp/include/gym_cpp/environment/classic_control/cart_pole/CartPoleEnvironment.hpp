
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>

#include <gym_cpp/environment/Environment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"


#include <gym_cpp/Net/PolicyNet.hpp>
class testPolicyNet : public PolicyNet{
public:
  testPolicyNet(int64_t N,int64_t M,int64_t H):
    PolicyNet(N,M,H)
  {
  }

  virtual void getActions(double* actions, int robotNumber) override{
    PolicyNet::getActions( actions, robotNumber);
    for(int i = 0 ; i< robotNumber ;i++){
      for(int j = 0 ; j< outputSize_ ;j++){
        if(*(actions+ outputSize_*i +j)  >0){
          *(actions+ outputSize_*i +j)  = 1 ;
        }else{
          *(actions+ outputSize_*i +j)  = 0 ;
        }
      }
    }
  }

  virtual void getMeanActions(double* actions, int robotNumber) override{
    PolicyNet::getMeanActions( actions, robotNumber);
    for(int i = 0 ; i< robotNumber ;i++){
      for(int j = 0 ; j< outputSize_ ;j++){
        if(*(actions+ outputSize_*i +j)  >0){
          *(actions+ outputSize_*i +j)  = 1 ;
        }else{
          *(actions+ outputSize_*i +j)  = 0 ;
        }
      }
    }
  }

};

namespace gym{
  class CartPoleEnvironment:public Environment{
    public:
      CartPoleEnvironment():
        Environment()
      {
        gravity_ = 9.8;
        massCart_ = 1.0;
        massPole_ = 0.1;
        totalMass_ = (massPole_ + massCart_);
        length_ = 0.5 ;// actually half the pole's length
        poleMassLength_ = (massPole_ * length_);
        forceMag_ = 10.0;
        tau_ = 0.02  ;// seconds between state updates
        thThresholdRadians_ = 12 * 2 * M_PI / 360;
        xThreshold_ = 2.4;
      }

      ~CartPoleEnvironment(){}

      bool step(Eigen::VectorXd& newState, double& reward, Eigen::VectorXd u) override{
        u_ = (int) u[0] ;
        if(u_!=0 && u_!=1){
          std::cout << "invalid action" <<std::endl;
          return false;
        }

        double force = (u_) ? forceMag_ : -forceMag_;

        double cosTh = cos(th_);
        double sinTh = sin(th_);
        double temp = (force + poleMassLength_ * thDot_ * thDot_ * sinTh) / totalMass_;
        double thAcc = (gravity_ * sinTh - cosTh* temp)
                     / (length_ * (4.0/3.0 - massPole_ * cosTh * cosTh / totalMass_));
        double xAcc  = temp - poleMassLength_ * thAcc * cosTh / totalMass_;
        // Euler
        x_  = x_ + tau_ * xDot_;
        xDot_ = xDot_ + tau_ * xAcc;
        th_ = th_ + tau_ * thDot_;
        thDot_ = thDot_ + tau_ * thAcc;

        bool done =  x_ < -xThreshold_ || x_ > xThreshold_ || th_ < -thThresholdRadians_
              || th_ > thThresholdRadians_;

        reward =  (done) ? 1 : 0;

        newState = Eigen::VectorXd::Zero(4);
        newState[0] = x_;
        newState[1] = xDot_;
        newState[2] = th_;
        newState[3] = thDot_;

        // TODO : implement done version
        return false;
      }

      Eigen::VectorXd reset() override{
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        x_ = 0.05 * distribution(generator);
        xDot_ = 0.05* distribution(generator);
        th_ = 0.05 * distribution(generator);
        thDot_ = 0.05 * distribution(generator);
        u_ = 0;
        Eigen::VectorXd state = Eigen::VectorXd::Zero(4);
        state[0] = x_;
        state[1] = xDot_;
        state[2] = th_;
        state[3] = thDot_;

        return state;
      }

      void close(){

      }

      void render() override{
        cv::Mat img(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));


        cv::Point center = cv::Point(x_/6*500+250, 350 );


        cv::Point edge = cv::Point((x_+length_*sin(th_))/6*500+250, 250-(-100+length_*cos(th_))/6*500 );

        if(edge.x<500 && edge.x>0 && center.x<500 && center.x>0  ){
          cv::line(img, center, edge, cv::Scalar(76.5,76.5,204.), 22, 8, 0 );
          cv::circle(img, center,5, cv::Scalar(0,0,0),CV_FILLED, 8,0);
        }
        cv::imshow("Pendulum-v0", img);
        cv::waitKey(10);
      }

      cv::Mat  renderVideo() override{
        cv::Mat img(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));


        cv::Point center = cv::Point(x_/6*500+250, 350 );


        cv::Point edge = cv::Point((x_+length_*sin(th_))/6*500+250, 350-(length_*cos(th_))/6*500 );

        if(edge.x<500 && edge.x>0 && center.x<500 && center.x>0  ){
          cv::line(img, center, edge, cv::Scalar(76.5,76.5,204.), 22, 8, 0 );
          cv::circle(img, center,5, cv::Scalar(0,0,0),CV_FILLED, 8,0);
        }
        cv::imshow("Pendulum-v0", img);
        cv::waitKey(10);
        return img;
      }

      int getStateSize() override{
        return  4;
      }

      int getActionSize() override{
        return  1;
      }

    protected:
      double gravity_ ;
      double massCart_ ;
      double massPole_ ;
      double totalMass_ ;
      double length_  ;
      double poleMassLength_;
      double forceMag_ ;
      double tau_   ;// seconds between state updates
      double thThresholdRadians_ ;
      double xThreshold_ ;

      double x_;
      double xDot_;
      double th_;
      double thDot_;

      int u_;

      double step_;


  };
}// namespace gym
