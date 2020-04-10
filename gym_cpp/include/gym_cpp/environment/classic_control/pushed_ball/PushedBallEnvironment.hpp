
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>

#include <gym_cpp/environment/Environment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace gym{
  class PushedBallEnvironment:public Environment{
    public:
      PushedBallEnvironment(double timeStep = 0.01):
        Environment()
      {
        maxPosition_ = 2.0;
        maxSpeed_ = 5.0;
        maxAction_ = 5.0;
        timeStep_ = timeStep;
      }

      ~PushedBallEnvironment(){}

      bool step(Eigen::VectorXd& newState, double& reward, Eigen::VectorXd u) override{

        u_ = u ;
        if(u_.norm()> maxAction_)
          u_ = u_.normalized() * maxAction_ ;


        velocity_ += timeStep_*u_;
        if(velocity_.norm()> maxSpeed_)
          velocity_ = velocity_.normalized() * maxSpeed_ ;

        position_+= timeStep_*velocity_;
        for(int i = 0; i<2 ;i++){
          if(position_[i]> maxPosition_)
            position_[i] = maxPosition_;
          else if (position_[i]< -maxPosition_)
            position_[i] = -maxPosition_ ;
        }

        reward = - 1.0*(desiredPosition_-position_).squaredNorm()
                 - 0.01*(u_).squaredNorm();

        Eigen::Vector2d sensingDirection;
        if(u_.norm()!=0){
          sensingDirection = u_.normalized();
        }
        double projection = sensingDirection.dot(desiredPosition_-position_);

        newState = Eigen::VectorXd::Zero(6);
        newState.segment(0,3) = newState.segment(3,3);
        newState.segment(3,2) = sensingDirection;
        newState[5] = projection;

        return false;
      }


      Eigen::VectorXd reset() override{
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        for(int i =0 ; i<2; i++){
          position_[i] = distribution(generator);
          desiredPosition_[i] = distribution(generator);
        }
        velocity_ = Eigen::Vector2d::Zero();
        u_ = Eigen::Vector2d::Zero();

        Eigen::VectorXd state = Eigen::VectorXd::Zero(6);

        Eigen::Vector2d sensingDirection = Eigen::Vector2d::Zero();
        for(int i =0 ; i<2; i++){
          sensingDirection[i] = distribution(generator);
        }
        sensingDirection = sensingDirection.normalized();
        double projection = sensingDirection.dot(desiredPosition_-position_);
        state.segment(0,2) = sensingDirection;
        state[2] = projection;

        for(int i =0 ; i<2; i++){
          sensingDirection[i] = distribution(generator);
        }
        sensingDirection = sensingDirection.normalized();
        projection = sensingDirection.dot(desiredPosition_-position_);
        state.segment(3,2) = sensingDirection;
        state[5] = projection;

        return state;
      }

      void close(){

      }

      void render() override{
        // cv::Mat img(640, 640, CV_8UC3, cv::Scalar(1, 1, 1));
        //
        // cv::Point ballpos = cv::Point(position_[0]/4*640+320, 320- position_[1]/4*640 );
        // cv::circle(img, ballpos,5
        //           , cv::Scalar(255,255,255),CV_FILLED, 8,0);
        // cv::Point despos = cv::Point(desiredPosition_[0]/4*640+320, 320- desiredPosition_[1]/4*640 );
        // cv::circle(img, despos,5
        //           , cv::Scalar(0,255,0),CV_FILLED, 8,0);
        //
        // cv::Point arrowHead = cv::Point(
        //     (position_[0]+velocity_[0]*2)/4*640+320,
        //     320-( position_[1]+velocity_[1]*2)/4*640 );
        //
        // cv::arrowedLine(img, ballpos,arrowHead
        //           , cv::Scalar(255,255,255),1, 8,0,0.1);
        //
        //
        // cv::imshow("PushedBall", img);
        // cv::waitKey(100);
      }

      cv::Mat renderVideo() override{
        int size = 500 ;

        cv::Mat img(size, size, CV_8UC3, cv::Scalar(1, 1, 1));


        std::cout << "________________" << std::endl;
        cv::Point ballpos = cv::Point(position_[0]/4*size+size/2, size/2- position_[1]/4*size );
        cv::circle(img, ballpos,5
                  , cv::Scalar(255,255,255),CV_FILLED, 8,0);
        cv::Point despos = cv::Point(desiredPosition_[0]/4*size+size/2, size/2- desiredPosition_[1]/4*size );
        cv::circle(img, despos,5
                  , cv::Scalar(0,255,0),CV_FILLED, 8,0);

        // cv::Point arrowHead = cv::Point(
        //     (position_[0]+velocity_[0]*2)/4*size+size/2,
        //     size/2-( position_[1]+velocity_[1]*2)/4*size );
        //
        // cv::arrowedLine(img, ballpos,arrowHead
        //           , cv::Scalar(255,255,255),1, 8,0,0.1);

        cv::imshow("PushedBall", img);
        cv::waitKey(100);
        return img;
      }

      int getStateSize() override{
        return  6;
      }

      int getActionSize() override{
        return  2;
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


  };
}// namespace gym
