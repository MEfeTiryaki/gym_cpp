#pragma once

#include <iostream>
#include <csignal>
#include <string>
#include <chrono>
#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <torch/torch.h>
#include "opencv2/opencv.hpp"


#include "gym_cpp/environment/Environment.hpp"
#include "gym_cpp/environment/ParallelEnvironment.hpp"
#include "trpo/model/StochasticPolicyNetBase.hpp"

void signalHandler(int signum ) {
   std::cout << "Closing.\n";
   exit(signum);
}

void play( gym::Environment& env
          , StochasticPolicyNetBase& policyNet
          ,int testNumber = 10
          ,int trajectoryLength = 200
          ,bool save=false
          ,std::string saveDirectory = "."
          ,std::string prefix = ""){

  cv::VideoWriter video(saveDirectory+"/"+prefix+"_test.avi"
        ,CV_FOURCC('M','J','P','G'),100, cv::Size(640,640));

  double* statePointer = (double*) malloc(sizeof(double)*env.getStateSize());
  double* actionPointer = (double*) malloc(sizeof(double)*env.getActionSize());
  double* rewardPointer = (double*) malloc(sizeof(double));

  Eigen::VectorXd rewardSum = Eigen::VectorXd::Zero(testNumber);
  Eigen::VectorXd state = Eigen::VectorXd::Zero(env.getStateSize());
  Eigen::VectorXd action = Eigen::VectorXd::Zero(env.getActionSize());

  double reward ;
  for(int i= 0 ;i <testNumber; i++){
    state =  env.reset();
    double reward = 0;
    for( int t =0 ; t< trajectoryLength ;t++){
      // Get Action
      for(int i_s =0 ; i_s<env.getStateSize();i_s++)
        *(statePointer+i_s) = state[i_s];
      policyNet.calculateAction(statePointer , 1,  env.getStateSize());
      policyNet.getMeanActions(actionPointer, 1);

      for(int i_a =0 ; i_a<env.getActionSize();i_a++)
        action[i_a] = *(actionPointer+i_a);
      // STEP
      bool done = env.step(state,reward,action);
      rewardSum[i] += reward;
      if (done){
        break;
      }
      env.render();
      if(save){
         video.write(env.renderVideo());
      }
    }
  }
  Eigen::VectorXd s = rewardSum - Eigen::VectorXd::Ones(testNumber)*rewardSum.mean();
  double std_dev = std::sqrt(s.dot(s)/(testNumber) );
  std::cout << "\t loss | mean : " << rewardSum.mean()<<" / std : "
    << std_dev << std::endl;

  free(statePointer);
  free(actionPointer);
  free(rewardPointer);
}


/*!
  return average cummulative reward
*/
double calculateACR( gym::Environment& env
                    , StochasticPolicyNetBase& policyNet
                    , int testNumber = 10
                    , int trajectoryLength = 200
                    , bool debug = false){
  double* statePointer = (double*) malloc(sizeof(double)*env.getStateSize());
  double* actionPointer = (double*) malloc(sizeof(double)*env.getActionSize());
  double* rewardPointer = (double*) malloc(sizeof(double));

  Eigen::VectorXd rewardSum = Eigen::VectorXd::Zero(testNumber);
  Eigen::VectorXd state = Eigen::VectorXd::Zero(env.getStateSize());
  Eigen::VectorXd action = Eigen::VectorXd::Zero(env.getActionSize());

  double reward ;
  for(int i= 0 ;i <testNumber; i++){
    state =  env.reset();
    double reward = 0;
    for( int t =0 ; t< trajectoryLength ;t++){
      // Get Action
      for(int i_s =0 ; i_s<env.getStateSize();i_s++)
        *(statePointer+i_s) = state[i_s];
      policyNet.calculateAction(statePointer , 1,  env.getStateSize());
      policyNet.getMeanActions(actionPointer, 1);

      for(int i_a =0 ; i_a<env.getActionSize();i_a++)
        action[i_a] = *(actionPointer+i_a);
      // STEP
      bool done = env.step(state,reward,action);
      rewardSum[i] += reward;
      if (done){
        break;
      }
    }
  }
  if(debug){
    Eigen::VectorXd s = rewardSum
            - Eigen::VectorXd::Ones(testNumber)*rewardSum.mean();
    double std_dev = std::sqrt(s.dot(s)/(testNumber) );
    std::cout << "\tACR | mean : " << rewardSum.mean()<<" / std : "<< std_dev;
  }
  free(statePointer);
  free(actionPointer);
  free(rewardPointer);
  return rewardSum.mean();
}

/*!
  return average cummulative reward
*/
double calculateACR( gym::ParallelEnvironment& env
                    , StochasticPolicyNetBase& policyNet
                    , int trajectoryLength = 200
                    , bool debug = false){
  int testNumber = env.getRobotNumber();
  double* statePointer = (double*) malloc(sizeof(double)*testNumber*env.getStateSize());
  double* actionPointer = (double*) malloc(sizeof(double)*testNumber*env.getActionSize());
  double* rewardPointer = (double*) malloc(sizeof(double)*testNumber);

  Eigen::VectorXd rewardSum = Eigen::VectorXd::Zero(testNumber);
  env.reset(statePointer);
  for(int t = 0; t< trajectoryLength; t++ ){
    // STEP
    policyNet.calculateAction(statePointer,testNumber,env.getStateSize());
    policyNet.getActions(actionPointer, testNumber);
    bool done = env.step(statePointer,rewardPointer,actionPointer);
    for(int i=0;i<testNumber;i++)
      rewardSum[i] += *(rewardPointer+i);
  }
  if(debug){
    Eigen::VectorXd s = rewardSum
            - Eigen::VectorXd::Ones(testNumber)*rewardSum.mean();
    double std_dev = std::sqrt(s.dot(s)/(testNumber) );
    std::cout << "\tACR | mean : " << rewardSum.mean()<<" / std : "<< std_dev;
  }
  free(statePointer);
  free(actionPointer);
  free(rewardPointer);
  return rewardSum.mean();
}

/*!
  return average terminal reward
*/
double calculateATR( gym::Environment& env
                    , StochasticPolicyNetBase& policyNet
                    , int testNumber = 10
                    , int trajectoryLength = 200
                    , bool debug = false){
  double* statePointer = (double*) malloc(sizeof(double)*env.getStateSize());
  double* actionPointer = (double*) malloc(sizeof(double)*env.getActionSize());
  double* rewardPointer = (double*) malloc(sizeof(double));

  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(testNumber);
  Eigen::VectorXd state = Eigen::VectorXd::Zero(env.getStateSize());
  Eigen::VectorXd action = Eigen::VectorXd::Zero(env.getActionSize());

  double reward ;
  for(int i= 0 ;i <testNumber; i++){
    state =  env.reset();
    double reward = 0;
    for( int t =0 ; t< trajectoryLength ;t++){
      // Get Action
      for(int i_s =0 ; i_s<env.getStateSize();i_s++)
        *(statePointer+i_s) = state[i_s];
      policyNet.calculateAction(statePointer , 1,  env.getStateSize());
      policyNet.getMeanActions(actionPointer, 1);

      for(int i_a =0 ; i_a<env.getActionSize();i_a++)
        action[i_a] = *(actionPointer+i_a);
      // STEP
      bool done = env.step(state,reward,action);

      if (done){
        break;
      }
    }
    rewards[i] = reward;
  }
  if(debug){
    Eigen::VectorXd s = rewards
            - Eigen::VectorXd::Ones(testNumber)*rewards.mean();
    double std_dev = std::sqrt(s.dot(s)/(testNumber) );
    std::cout << "\tATR | mean : " << rewards.mean()<<" / std : "<< std_dev;
  }
  free(statePointer);
  free(actionPointer);
  free(rewardPointer);
  return rewards.mean();
}

/*!
return average terminal reward
*/
double calculateATR( gym::ParallelEnvironment& env
                    , StochasticPolicyNetBase& policyNet
                    , int trajectoryLength = 200
                    , bool debug = false){
  int testNumber = env.getRobotNumber();
  double* statePointer = (double*) malloc(sizeof(double)*testNumber*env.getStateSize());
  double* actionPointer = (double*) malloc(sizeof(double)*testNumber*env.getActionSize());
  double* rewardPointer = (double*) malloc(sizeof(double)*testNumber);

  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(testNumber);
  env.reset(statePointer);
  for(int t = 0; t< trajectoryLength; t++ ){
    // STEP
    policyNet.calculateAction(statePointer,testNumber,env.getStateSize());
    policyNet.getActions(actionPointer, testNumber);
    bool done = env.step(statePointer,rewardPointer,actionPointer);
  }
  for(int i=0;i<testNumber;i++)
    rewards[i] = *(rewardPointer+i);

  if(debug){
    Eigen::VectorXd s = rewards
            - Eigen::VectorXd::Ones(testNumber)*rewards.mean();
    double std_dev = std::sqrt(s.dot(s)/(testNumber) );
    std::cout << "\tATR | mean : " << rewards.mean()<<" / std : "<< std_dev;
  }
  free(statePointer);
  free(actionPointer);
  free(rewardPointer);
  return rewards.mean();
}

torch::Tensor loadParameterCsv(std::string filename){
  std::fstream fin;
  fin.open(filename, std::ios::in);

  if (!fin.good()){
    std::cout << "Error : " << filename << " doesn't exist!! "<< std::endl;
    return torch::zeros({0},at::kDouble);
  }

  std::vector<double> row = std::vector<double>();
  std::string line, word, temp;
  row.clear();
  getline(fin, line);
  std::stringstream s(line);
  while (getline(s, word, ',')) {
   row.push_back(std::stod(word));
  }
  torch::Tensor param =  torch::zeros({(long)row.size()},at::kDouble) ;

  for(int i =0 ; i < row.size(); i++){
    param[i] =  row[i];
  }

  return param;
}

void saveParameterCsv(torch::Tensor p,std::string filename){
   std::fstream fin;
   fin.open(filename, std::ios::out);

   for(int i = 0; i<p.sizes()[0];i++){
     fin<< p[i].item<double>() << ",";
   }
   fin<<  "\n";

}

torch::Tensor loadTensorCsv(std::string filename){
   std::fstream fin;
   fin.open(filename, std::ios::in);


   std::vector<std::vector<double>> rows = std::vector<std::vector<double>>();
   std::string line, word, temp;
   while(std::getline(fin, line)){
     std::vector<double> row = std::vector<double>();
     row.clear();
     // getline(fin, line);
     std::stringstream s(line);
     while (getline(s, word, ',')) {
         row.push_back(std::stod(word));
     }
     rows.push_back(row);
   }
   torch::Tensor t =  torch::zeros({(long)rows.size(),(long)rows[0].size()},at::kDouble) ;
   for(int i =0 ; i < rows.size(); i++){
     for(int j =0 ; j < rows[0].size(); j++){
       t[i][j] =  rows[i][j];
     }
   }

   return t;
}

class LearningLogger{
  public:
    LearningLogger(std::string filename){
      filename_ = filename;
      clean();
    }

    ~LearningLogger(){}

    void cleanFile(){
      std::fstream fin;
      fin.open(filename_, std::ios::out);
      fin.close();
    }

    void addLog(std::string x){
      if(line_!=""){
        line_+=",";
      }
      line_+=x;
    }
    void addLog(char* x ){ addLog(std::string(x));}
    void addLog(bool x ){ addLog(std::to_string(x));}
    void addLog(int x ){ addLog(std::to_string(x));}
    void addLog(unsigned x ){ addLog(std::to_string(x));}
    void addLog(long x ){ addLog(std::to_string(x));}
    void addLog(float x ){ addLog(std::to_string(x));}
    void addLog(double x ){ addLog(std::to_string(x));}

    void write(){
      std::fstream fin;
      fin.open(filename_, std::ios::app);
      fin<< line_<< "\n";
      fin.close();
      clean();
    }

  protected:
    void clean(){
      line_="";
    }
  private:
    std::string filename_;
    std::string line_;
};
