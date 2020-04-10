#pragma once

#include <Eigen/Dense>

class Spaces{
  public:
    Spaces(){}

    ~Spaces(){}

    Eigen::VectorXd sample(){
      return Eigen::VectorXd::Zero(0);
    }
  protected:

}
