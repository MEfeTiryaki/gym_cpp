
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>

#include <gym_cpp/environment/Environment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace gym{
  class PendulumEnvironment:public Environment{
    public:
      PendulumEnvironment():
        Environment()
      {
        maxSpeed_ = 8.0;
        maxTorque_ = 2.0;
        dt_= 0.05;
        g_ = 10.0;
        m_ = 1.0;
        l_ = 1.0;
      }

      ~PendulumEnvironment(){}

      bool step(Eigen::VectorXd& newState, double& reward, Eigen::VectorXd u) override{
        u_ = u[0] ;
        if(u_> maxTorque_)
          u_ = maxTorque_ ;
        else if (u_< -maxTorque_)
          u_ = -maxTorque_ ;

        reward = -( pow(angle_normalize(th_),2)
                  + 0.1*pow(thdot_,2)
                  + 0.001*pow(u_,2));

        double newthdot = thdot_
                        + (-3*g_/(2*l_)
                        * sin(th_ + M_PI)
                        + 3.0/(m_*pow(l_,2))*u_) * dt_;
        double newth = th_ + newthdot*dt_;
        if(newthdot> maxSpeed_)
          newthdot = maxSpeed_ ;
        else if (newthdot< -maxSpeed_)
          newthdot = -maxSpeed_ ;
        newState = getObservation(newth, newthdot);
        th_ = newth;
        thdot_ = newthdot;

        step_++;
        if(step_==200){
          return true;
        }
        return false;
        //return self._get_obs(), -costs, False, {}
      }

      Eigen::VectorXd reset() override{
        step_ = 0;
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        th_ = M_PI * distribution(generator);
        thdot_ = 1 * distribution(generator);
        u_ = 0;
        return getObservation(th_,thdot_);
      }

      void setInitialState(double th,double thdot){
        th_ = th;
        thdot_ = thdot;
      }

      void close(){

      }

      void render() override{
        image_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Point edge = cv::Point(l_*sin(th_)/6*640+320, 320-l_*cos(th_)/4*640 );
        cv::line(image_, cv::Point(320, 320), edge, cv::Scalar(76.5,76.5,204.), 22, 8, 0 );
        cv::circle(image_, cv::Point(320, 320),5, cv::Scalar(0,0,0),CV_FILLED, 8,0);

        cv::imshow("Pendulum-v0", image_);
        cv::waitKey(10);
      }

      cv::Mat renderVideo() override{
        return image_;
      }

      int getStateSize() override{
        return  3;
      }

      int getActionSize() override{
        return  1;
      }
    protected:
      Eigen::Vector3d getObservation(double theta,double thetadot){
        return Eigen::Vector3d(cos(theta), sin(theta), thetadot);
      }
      double angle_normalize(double x){

        while(x>M_PI)
          x -= 2*M_PI;
        while(x<-M_PI)
          x += 2*M_PI;
        return x;
        // return (std::fmod( x+M_PI , 2*M_PI ) - M_PI);
      }

    protected:
      double maxSpeed_;
      double maxTorque_;
      double dt_;
      double g_;
      double m_;
      double l_;

      double th_;
      double thdot_;

      double u_;

      double step_;

      cv::Mat image_;

  };
}// namespace gym
