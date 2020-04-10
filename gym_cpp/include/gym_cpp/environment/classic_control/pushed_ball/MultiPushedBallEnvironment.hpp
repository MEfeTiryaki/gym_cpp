
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>


#include <gym_cpp/environment/ParallelEnvironment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace gym{
  class MultiPushedBallEnvironment:public ParallelEnvironment{
    public:
      MultiPushedBallEnvironment(int robotNumber, double maxAction=5.0, double maxPosition = 2.0,double maxSpeed = 5.0, double timeStep = 0.01):
        ParallelEnvironment()
      {
        robotNumber_ = robotNumber;
        maxPosition_ = maxPosition;
        maxSpeed_ = maxSpeed;
        maxAction_ = maxAction;
        timeStep_ = timeStep;

        position_ = (double*) malloc(sizeof(double)* robotNumber_*2);
        desiredPosition_ = (double*) malloc(sizeof(double)* robotNumber_*2);
        velocity_ = (double*) malloc(sizeof(double)* robotNumber_*2);
        u_ = (double*) malloc(sizeof(double)* robotNumber_*2);

      }

      ~MultiPushedBallEnvironment(){}


      bool step(double* newState, double* reward, double* u) override{
        for(int i = 0; i<robotNumber_;i++){
          double u_magnitute = 0;
          for(int j = 0; j<2 ;j++){
            *(u_+2*i+j) = *(u+2*i+j);
            u_square += *(u_+2*i+j)* *(u_+2*i+j);
          }
          u_magnitute = sqrt(u_square);

          if(u_magnitute>maxAction_){
            u_magnitute = maxAction_;
          }
          // Normalize the input
          for(int j = 0; j<2 ;j++){
            *(u_+2*i+j) = *(u_+2*i+j)/u_magnitute;
          }

          for(int j = 0; j<2 ;j++){
            *(velocity_+2*i+j) += timeStep_* *(u_+2*i+j) * u_magnitute;
            *(position_+2*i+j) += timeStep_* *(velocity_+2*i+j);

            if(*(velocity_+2*i+j)> maxSpeed_)
              *(velocity_+2*i+j) = maxSpeed_ ;
            else if (*(velocity_+2*i+j)< -maxSpeed_)
              *(velocity_+2*i+j) = -maxSpeed_;

            if(*(position_+2*i+j)> maxPosition_)
              *(position_+2*i+j) = maxPosition_ ;
            else if (*(position_+2*i+j)< -maxPosition_)
              *(position_+2*i+j) = -maxPosition_;

            double squareNormError = (*(desiredPosition_+2*i) - *(position_+2*i))
                         * (*(desiredPosition_+2*i) - *(position_+2*i) )
                         + (*(desiredPosition_+2*i+1) - *(position_+2*i+1))
                         * (*(desiredPosition_+2*i+1) - *(position_+2*i+1));
                         
            *(reward+i) = - 1.0* squareNormError - 0.01 * u_magnitute*u_magnitute;

            // No need to be in for j
            *(newState+6*i)   = *(newState+6*i+3);
            *(newState+6*i+1) = *(newState+6*i+4);
            *(newState+6*i+2) = *(newState+6*i+5);

            *(newState+6*i+3) = -*(velocity_+2*i+1) ;
            *(newState+6*i+4) = -*(velocity_+2*i+1) ;
            *(newState+6*i+5) = -*(velocity_+2*i+1) ;
          }
        }

        return false;
      }

      void reset(double* state) override{
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        double initSet = maxPosition_/2;
        if(initSet>1){
          initSet = 1;
        }
        for(int i = 0; i<robotNumber_;i++){
          for(int j = 0; j<2 ;j++){
            *(position_+2*i+j) = initSet * distribution(generator);
            *(desiredPosition_+2*i+j) = initSet * distribution(generator);
            *(velocity_+2*i+j) = 0;
            *(u_+2*i+j) = 0;
            *(state+4*i)   = *(desiredPosition_+2*i) - *(position_+2*i) ;
            *(state+4*i+1) = *(desiredPosition_+2*i+1) - *(position_+2*i+1) ;
            *(state+4*i+2) = -*(velocity_+2*i) ;
            *(state+4*i+3) = -*(velocity_+2*i+1) ;
          }
        }

      }

      void close(){

      }

      void render() override{
        cv::Mat img(640, 640, CV_8UC3, cv::Scalar(1, 1, 1));

        cv::Point ballpos = cv::Point(position_[0]/4*640+320, 320- position_[1]/4*640 );
        cv::circle(img, ballpos,5
                  , cv::Scalar(255,255,255),CV_FILLED, 8,0);
        cv::Point despos = cv::Point(desiredPosition_[0]/4*640+320, 320- desiredPosition_[1]/4*640 );
        cv::circle(img, despos,5
                  , cv::Scalar(0,255,0),CV_FILLED, 8,0);

        cv::imshow("MovingBall", img);
        cv::waitKey(10);
      }


      int getStateSize() override{
        return  4;
      }

      int getActionSize() override{
        return  2;
      }
      int getRobotNumber() override {
        return robotNumber_;
      }
    protected:

      int robotNumber_;

      double maxPosition_;
      double maxSpeed_;
      double maxAction_;

      double timeStep_;

      double* position_;
      double* desiredPosition_;
      double* velocity_;
      double* u_;


  };
}// namespace gym
