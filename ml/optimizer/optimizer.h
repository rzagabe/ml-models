// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#ifndef ML_OPTIMIZER_OPTIMIZER_H_
#define ML_OPTIMIZER_OPTIMIZER_H_

#include "ml/optimizer/optimizer.pb.h"

#include <iostream>
#include <string>
#include <memory>

#include "Eigen/Core"
#include "glog/logging.h"
#include "ml/factory.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {

class Optimizer : public FactoryBase {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  Optimizer() {}

  virtual ~Optimizer() {}

  virtual bool Optimize(const Matrix& x, const Vector& y, int min_iter,
                        int max_iter, double eps, double gamma, Vector* w) = 0;
};

#define REGISTER_OPTIMIZER(TYPE) REGISTER_CONCRETE(TYPE, TYPE, Optimizer)

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OPTIMIZER_OPTIMIZER_H_
