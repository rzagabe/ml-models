// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Provides an ordinary squared loss function implementation.

#ifndef ML_OPTIMIZER_SQUARED_LOSS_H_
#define ML_OPTIMIZER_SQUARED_LOSS_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {

class SquaredLoss : public LossFunction {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  SquaredLoss() {}

  virtual ~SquaredLoss() {}

  virtual bool eval_f(const Matrix& x, const Vector& y, const Vector& w,
                      double* f) {
    CHECK(f != nullptr);
    Matrix tmp = (y - (x * w)).transpose() * (y - (x * w));
    *f = tmp.sum() / 2;
    return true;
  }

  virtual bool eval_gradient_f(const Matrix& x, const Vector& y,
                               const Vector& w, Vector* gradient_f) {
    CHECK(gradient_f != nullptr);
    *gradient_f = x.transpose() * ((x * w) - y);
    return true;
  }

  virtual bool intermediate_callback(const Matrix& x, const Vector& y,
                                     int cur_iter, int min_iter, int max_iter,
                                     double eps, double* gamma, Vector* w) {
    return true;
  }
};

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OPTIMIZER_SQUARED_LOSS_H_
