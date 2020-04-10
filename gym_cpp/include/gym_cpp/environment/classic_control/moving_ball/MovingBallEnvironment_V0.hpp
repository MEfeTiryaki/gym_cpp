
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>

#include <gym_cpp/environment/Environment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace gym{
  class MovingBallEnvironment_V0:public Environment{
    public:
      MovingBallEnvironment_V0( double maxAction=5.0, double maxPosition = 2.0,double maxSpeed = 5.0,double timeStep = 0.01):
        Environment()
      {
        maxPosition_ = maxPosition;
        maxSpeed_ = maxSpeed;
        maxAction_ = maxAction;
        timeStep_ = timeStep;
      }

      ~MovingBallEnvironment_V0(){}

      bool step(Eigen::VectorXd& newState, double& reward, Eigen::VectorXd u) override{
        u_ = u ;
        for(int i = 0; i<2 ;i++){
          if(u_[i]> maxAction_)
            u_[i] = maxAction_ ;
          else if (u_[i]< -maxAction_)
            u_[i] = -maxAction_ ;
        }


        velocity_ += timeStep_*u_;
        for(int i = 0; i<2 ;i++){
          if(velocity_[i]> maxSpeed_)
            velocity_[i] = maxSpeed_ ;
          else if (velocity_[i]< -maxSpeed_)
            velocity_[i] = -maxSpeed_;
        }
        position_+= timeStep_*velocity_;
        for(int i = 0; i<2 ;i++){
          if(position_[i]> maxPosition_)
            position_[i] = maxPosition_;
          else if (position_[i]< -maxPosition_)
            position_[i] = -maxPosition_ ;
        }

        reward = - 1.0*(desiredPosition_-position_).squaredNorm()
                 - 0.01*(u_).squaredNorm();

        newState = Eigen::VectorXd::Zero(4);
        newState.segment(0,2) = desiredPosition_-position_;
        newState.segment(2,2) = -velocity_;

        return false;
      }


      Eigen::VectorXd reset() override{
        std::random_device rd;
        std::default_random_engine generator(rd());
          std::uniform_real_distribution<double> distribution(-1.0,1.0);
        double initSet = maxPosition_/10;
        for(int i =0 ; i<2; i++){
          position_[i] = initSet * distribution(generator);
          desiredPosition_[i] = initSet * distribution(generator);
        }
        velocity_ = Eigen::Vector2d::Zero();
        u_ = Eigen::Vector2d::Zero();

        Eigen::VectorXd state = Eigen::VectorXd::Zero(4);
        state.segment(0,2) = desiredPosition_-position_;
        state.segment(2,2) = -velocity_;

        return state;
      }

      void setInitialState(Eigen::Vector2d position,
                           Eigen::Vector2d desiredPosition,
                           Eigen::Vector2d velocity){
        position_ = position;
        desiredPosition_= desiredPosition;
        velocity_ = velocity;
      }

      void close(){

      }

      void render() override{
        image_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(1, 1, 1));

        cv::Point ballpos = cv::Point(position_[0]/4*640+320, 320- position_[1]/4*640 );
        cv::circle(image_, ballpos,5
                  , cv::Scalar(255,255,255),CV_FILLED, 8,0);
        cv::Point despos = cv::Point(desiredPosition_[0]/4*640+320, 320- desiredPosition_[1]/4*640 );
        cv::circle(image_, despos,5
                  , cv::Scalar(0,255,0),CV_FILLED, 8,0);

        cv::Point arrowHead = cv::Point(
            (position_[0]+velocity_[0]*2)/4*640+320,
            320-( position_[1]+velocity_[1]*2)/4*640 );

        cv::arrowedLine(image_, ballpos,arrowHead
                  , cv::Scalar(255,255,255),1, 8,0,0.1);

        cv::imshow("MovingBall", image_);
        cv::waitKey(10);
      }

      cv::Mat renderVideo() override{
        return image_;
      }

      int getStateSize() override{
        return  4;
      }

      int getActionSize() override{
        return  2;
      }

    protected:

      double getHeight(double xs){
        return sin(3 * xs) * .45 + .55;
      }

    protected:
      double maxPosition_;
      double maxSpeed_;
      double maxAction_;

      double timeStep_;

      Eigen::Vector2d position_;
      Eigen::Vector2d desiredPosition_;
      Eigen::Vector2d velocity_;
      Eigen::Vector2d u_;

      cv::Mat image_;


  };
}// namespace gym
