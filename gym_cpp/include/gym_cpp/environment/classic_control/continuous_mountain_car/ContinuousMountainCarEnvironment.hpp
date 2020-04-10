
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>

#include <gym_cpp/environment/Environment.hpp>

namespace gym{
  class ContinuousMountainCarEnvironment:public Environment{
    public:
      ContinuousMountainCarEnvironment(double goalVelocity = 0.0):
        Environment()
      {
        minAction_ = -1.0;
        maxAction_ = 1.0;
        minPosition_ = -1.2;
        maxPosition_ = 0.6;
        maxSpeed_ = 0.07;
        goalPosition_ = 0.45;
        goalVelocity_ = goalVelocity;
        power_ = 0.0015;

      }

      ~ContinuousMountainCarEnvironment(){}

      bool step(Eigen::VectorXd& newState, double& reward, Eigen::VectorXd u)  override{
        u_ = u[0] ;
        if(u_> maxAction_)
          u_ = maxAction_ ;
        else if (u_< minAction_)
          u_ = minAction_ ;


        velocity_ += u_*power_- 0.0025* cos(3*position_);
        if(velocity_> maxSpeed_)
          velocity_ = maxSpeed_ ;
        else if (velocity_< -maxSpeed_)
          velocity_ = -maxSpeed_ ;

        position_+= velocity_;
        if(position_> maxPosition_)
          position_ = maxPosition_ ;
        else if (position_< minPosition_)
          position_ = minPosition_ ;

        bool done =  position_ >= goalPosition_ && velocity_ >= goalVelocity_;

        reward = 0;
        if (done){
          reward = 100.0;
        }
        reward -= 0.1*pow(u_,2);

        newState = Eigen::Vector2d(position_,velocity_);

        return done;
      }

      Eigen::VectorXd reset() override{
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        position_ = -0.6 +  0.2 * distribution(generator);
        velocity_ = 0.0;
        u_ = 0;
        return Eigen::Vector2d(position_,velocity_);
      }

      void setInitialState(double position,double velocity){
        position_ = position;
        velocity_ = velocity;
      }

      void close(){

      }

      int getStateSize() override{
        return  2;
      }

      int getActionSize() override{
        return  1;
      }

    protected:

      double getHeight(double xs){
        return sin(3 * xs) * .45 + .55;
      }

    protected:
      double minAction_;
      double maxAction_;
      double minPosition_;
      double maxPosition_;
      double maxSpeed_;
      double goalPosition_;
      double goalVelocity_;
      double power_ ;


      double position_;
      double velocity_;
      double u_;


  };
}// namespace gym
