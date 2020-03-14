
#include <Eigen/Dense>
#include <gym_cpp/environment/classic_control/PendulumEnvironment.hpp>
#include <iostream>
int main() {
  gym::PendulumEnvironment* env = new gym::PendulumEnvironment();

  double reward ;
  Eigen::VectorXd observation;
  for(int episodeId = 0; episodeId< 1; episodeId++ ){
    observation =  env->reset();
    reward = 0;
    for(int t = 0; t< 10; t++ ){
      std::cout << observation.transpose() << "|"<< reward << std::endl;
      Eigen::VectorXd action = Eigen::VectorXd::Zero(1);
      bool done = env->step(observation,reward,action);
      if(done)
        break;
    }
  }
}
