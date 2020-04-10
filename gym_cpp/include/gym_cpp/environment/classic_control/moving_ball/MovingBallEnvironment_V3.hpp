
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>


#include <gym_cpp/environment/ParallelEnvironment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace gym{
  class MovingBallEnvironment_V3:public ParallelEnvironment{
    public:
      MovingBallEnvironment_V3(int robotNumber,int stateHistoryLength = 1 , double maxAction=5.0, double maxPosition = 2.0,double maxSpeed = 5.0, double timeStep = 0.01):
        ParallelEnvironment()
      {
        robotNumber_ = robotNumber;
        stateHistoryLength_ = stateHistoryLength;
        stateSize_ = stateHistoryLength_*2;
        maxPosition_ = maxPosition;
        maxSpeed_ = maxSpeed;
        maxAction_ = maxAction;
        timeStep_ = timeStep;

        position_ = (double*) malloc(sizeof(double)* robotNumber_*2);
        desiredPosition_ = (double*) malloc(sizeof(double)* robotNumber_*2);
        velocity_ = (double*) malloc(sizeof(double)* robotNumber_*2);
        u_ = (double*) malloc(sizeof(double)* robotNumber_*2);

      }

      ~MovingBallEnvironment_V3(){}


      bool step(double* newState, double* reward, double* u) override{
        for(int i = 0; i<robotNumber_;i++){
          double* state = (double*) malloc(sizeof(double)* 2);

          for(int j = 0; j<2 ;j++){
            *(u_+2*i+j) = *(u+2*i+j);
            if(*(u_+2*i+j) > maxAction_)
              *(u_+2*i+j) = maxAction_ ;
            else if(*(u_+2*i+j)< -maxAction_)
              *(u_+2*i+j) = -maxAction_ ;

            *(velocity_+2*i+j) += timeStep_* *(u_+2*i+j);
            *(position_+2*i+j) += timeStep_* *(velocity_+2*i+j);

            if(*(velocity_+2*i+j)> maxSpeed_)
              *(velocity_+2*i+j) = maxSpeed_ ;
            else if (*(velocity_+2*i+j)< -maxSpeed_)
              *(velocity_+2*i+j) = -maxSpeed_;

            if(*(position_+2*i+j)> maxPosition_)
              *(position_+2*i+j) = maxPosition_ ;
            else if (*(position_+2*i+j)< -maxPosition_)
              *(position_+2*i+j) = -maxPosition_;

            *(state+j) = *(newState+stateSize_*i+(stateSize_-2)+j);

            for(int k=0;k<stateHistoryLength_-1;k++){
              *(newState+stateSize_*i+k*2+j)   = *(newState+stateSize_*i+(k+1)*2+j);
            }
            *(newState+stateSize_*i+(stateSize_-2)+j) =
                        *(desiredPosition_+2*i+j) - *(position_+2*i+j) ;
          }



          double squareNormError = *(newState+stateSize_*i+(stateSize_-2))
                                 * *(newState+stateSize_*i+(stateSize_-2))
                                 + *(newState+stateSize_*i+(stateSize_-2)+1)
                                 * *(newState+stateSize_*i+(stateSize_-2)+1);
          double squareNormVel = (*(newState+stateSize_*i+(stateSize_-2))-*(state))
                               * (*(newState+stateSize_*i+(stateSize_-2))-*(state))
                               + (*(newState+stateSize_*i+(stateSize_-2)+1)-*(state+1))
                               * (*(newState+stateSize_*i+(stateSize_-2)+1)-*(state+1)) ;
          double squareNormU = *(u_+2*i)  * *(u_+2*i)
                             + *(u_+2*i+1)* *(u_+2*i+1);


          *(reward+i) = - 1.0* squareNormError - 0.1*squareNormVel - 0.01 * squareNormU;

          free(state);
        }

        return false;
      }

      void reset(double* state) override{
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        double initSet = maxPosition_/10;

        for(int i = 0; i<robotNumber_;i++){
          for(int j = 0; j<2 ;j++){
            *(position_+2*i+j) = initSet * distribution(generator);
            *(desiredPosition_+2*i+j) = initSet * distribution(generator);
            *(velocity_+2*i+j) = initSet * distribution(generator);
            *(u_+2*i+j) = 0;
            for(int k=0;k<stateHistoryLength_;k++){
              *(state+stateSize_*i+k*2+j)   = *(desiredPosition_+2*i+j)
                                            - (*(position_+2*i+j)-(stateHistoryLength_-1-k)*timeStep_* *(velocity_+2*i+j)) ;
            }
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
        return  stateSize_;
      }

      int getActionSize() override{
        return  2;
      }
      int getRobotNumber() override {
        return robotNumber_;
      }
    protected:

      int robotNumber_;

      int stateSize_;
      int stateHistoryLength_;

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
