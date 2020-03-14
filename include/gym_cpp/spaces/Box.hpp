


#include <Eigen/Dense>

namespace gym{
  namespace spaces{
    class Box{
      public:
        Box(Eigen::VectorXd low, Eigen::VectorXd high):
          low_(low),
          high_(high)
        {}

        ~Box(){}

        Eigen::VectorXd sample(){
          return Eigen::VectorXd::Zero(0);
        }
      protected:
        Eigen::VectorXd low_;
        Eigen::VectorXd high_;
    };
  }
}
