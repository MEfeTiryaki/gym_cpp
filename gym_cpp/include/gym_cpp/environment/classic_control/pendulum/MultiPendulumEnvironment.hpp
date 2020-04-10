
#pragma once

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <thread>

#include <omp.h>

#include <gym_cpp/environment/ParallelEnvironment.hpp>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace gym{
  class MultiPendulumEnvironment:public ParallelEnvironment{
    public:
      MultiPendulumEnvironment(int robotNumber ):
        ParallelEnvironment()
      {
        cpuNum_ = std::thread::hardware_concurrency();

        robotNumber_= robotNumber;
        maxSpeed_ = 8.0;
        maxTorque_ = 2.0;
        dt_= 0.05;
        g_ = 10.0;
        m_ = 1.0;
        l_ = 1.0;

        th_ = (double*) malloc(sizeof(double)* robotNumber_);
        thdot_ = (double*) malloc(sizeof(double)* robotNumber_);
        u_ = (double*) malloc(sizeof(double)* robotNumber_);

      }

      ~MultiPendulumEnvironment(){}

      bool step(double* newState, double* reward, double* u) override{
        int i;
        #pragma omp parallel num_threads(cpuNum_)
        {
          #pragma omp for private(i)
            for( i = 0; i<robotNumber_;i++){
              *(u_+i) = *(u+i) ;
              if(*(u_+i)> maxTorque_)
                *(u_+i) = maxTorque_ ;
              else if (*(u_+i)< -maxTorque_)
                *(u_+i) = -maxTorque_ ;

              *(reward+i) = -( pow(angle_normalize(*(th_+i)),2)
                             + 0.1*pow(*(thdot_+i),2)
                             + 0.001*pow(*(u_+i),2));

              double newthdot = *(thdot_+i)
                              + (-3*g_/(2*l_)
                              * sin(*(th_+i) + M_PI)
                              + 3.0/(m_*pow(l_,2))* *(u_+i)) * dt_;
              double newth = *(th_+i) + newthdot*dt_;
              if(newthdot> maxSpeed_)
                newthdot = maxSpeed_ ;
              else if (newthdot< -maxSpeed_)
                newthdot = -maxSpeed_ ;

              *(newState+3*i) = cos(newth);
              *(newState+3*i+1) = sin(newth);
              *(newState+3*i+2) = newthdot;
              *(th_+i) = newth;
              *(thdot_+i) = newthdot;
            }
        }
        step_++;
        if(step_==200){
          return true;
        }
        return false;
        //return self._get_obs(), -costs, False, {}
      }

      void reset(double* state) override{
        step_ = 0;
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0,1.0);

        for(int i = 0; i<robotNumber_;i++){
          *(th_+i) = M_PI * distribution(generator);
          *(thdot_+i) = 1 * distribution(generator);
          *(u_ +i)= 0;
          *(state+3*i) = cos(*(th_+i));
          *(state+3*i+1) = sin(*(th_+i));
          *(state+3*i+2) = *(thdot_+i);
        }
      }


      void close(){

      }

      int getStateSize() override{
        return  3;
      }

      int getActionSize() override{
        return  1;
      }

      int getRobotNumber() override {
        return robotNumber_;
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
      int robotNumber_;
      int cpuNum_;

      double maxSpeed_;
      double maxTorque_;
      double dt_;
      double g_;
      double m_;
      double l_;

      double* th_;
      double* thdot_;
      double* u_;

      double step_;


  };
}// namespace gym
