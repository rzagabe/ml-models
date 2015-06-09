// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#ifndef ML_OPTIMIZER_LASSO_H_
#define ML_OPTIMIZER_LASSO_H_

#include "ml/optimizer/estimator.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace optimizer {

class Lasso : public Estimator {
 public:
  Lasso(const Matrix& x, const Vector& y) : x_(x), y_(y) {}

  virtual ~Lasso() {}

  virtual bool Init() { return true; }

  virtual bool eval_f(const Vector& w, double* f) {
    CHECK(f != nullptr);
    return true;
  }

  virtual bool eval_gradient_f(const Vector& w, Vector* gradient) {
    return true;
  }

  double l1_reg() { return l1_reg_; }
  void set_l1_reg(double l1_reg) { l1_reg_ = l1_reg; }

 private:
  // Regularization l1-norm.
  double l1_reg_;

  const Matrix& x_;

  const Vector& y_;
};

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OPTIMIZER_LASSO_H_
