#pragma once

#include <gym_cpp/environment/Environment.hpp>
namespace gym{
  class ParallelEnvironment: public Environment{
    public:
      ParallelEnvironment():
        Environment()
      {
      }

      ~ParallelEnvironment(){}

      virtual bool step(double* newState, double* reward, double* u){}

      virtual void reset(double* state){}

      virtual int getRobotNumber() {
        return 0;
      }
  };
}// namespace gym
