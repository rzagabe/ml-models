// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// L2-regularized loss function implementaion.

#ifndef ML_OPTIMIZER_RIDGE_LOSS_H_
#define ML_OPTIMIZER_RIDGE_LOSS_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "google/protobuf/message.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/optimizer.pb.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {

class RidgeLoss : public LossFunction {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  RidgeLoss() : l2_reg_(0.0001) {}

  virtual ~RidgeLoss() {}

  virtual bool eval_f(const Matrix& x, const Vector& y, const Vector& w,
                      double* f) {
    CHECK(f != nullptr);
    Matrix tmp = (y - (x * w)).transpose() * (y - (x * w));
    *f = (tmp.sum() / 2) + ((l2_reg_ / 2) * (w.dot(w)));
    return true;
  }

  virtual bool eval_gradient_f(const Matrix& x, const Vector& y,
                               const Vector& w, Vector* gradient_f) {
    CHECK(gradient_f != nullptr);
    *gradient_f = x.transpose() * ((x * w) - y) + (l2_reg_ * w);
    return true;
  }

  virtual bool intermediate_callback(const Matrix& x, const Vector& y,
                                     int cur_iter, int min_iter, int max_iter,
                                     double eps, double* gamma, Vector* w) {
    return true;
  }

  double l2_reg() const { return l2_reg_; }
  void set_l2_reg(double l2_reg) { l2_reg_ = l2_reg; }

 private:
  double l2_reg_;
};

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OPTIMIZER_RIDGE_LOSS_H_
