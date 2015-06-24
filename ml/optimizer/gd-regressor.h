// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Gradient Descent implementation.

#ifndef ML_OPTIMIZER_GD_REGRESSOR_H_
#define ML_OPTIMIZER_GD_REGRESSOR_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/optimizer.h"
#include "ml/optimizer/optimizer.pb.h"

namespace ml {
namespace optimizer {

class GDRegressor : public Optimizer {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  GDRegressor() : l2_reg_(0.0), cached_iterations_(0), cached_cost_(0.0) {}

  virtual ~GDRegressor() {}

  virtual bool Initialize(const OptimizerParameters& parameters) {
    l2_reg_ = parameters.l2_regularization();
    bt_line_search_ = parameters.bt_line_search();
    bt_line_search_alpha_ = parameters.bt_line_search_alpha();
    bt_line_search_beta_ = parameters.bt_line_search_beta();
    return true;
  }

  // Executes the given optimization settings.
  virtual bool Optimize(const Matrix& x, const Vector& y, int min_iter,
                        int max_iter, double eps, double gamma, Vector* w);

  void set_l2_reg(double l2_reg) { l2_reg_ = l2_reg; }
  double l2_reg() const { return l2_reg_; }

  int cached_iterations() const { return cached_iterations_; }
  double cached_cost() const { return cached_cost_; }

 private:
  // Cost function evaluation.
  double eval_f(const Matrix& x, const Vector& y, const Vector& w) {
    Matrix tmp = (y - (x * w)).transpose() * (y - (x * w));
    return (tmp.sum() / 2) + ((l2_reg_ / 2) * w.dot(w));
  }

  // Gradient of the error function.
  void eval_gradient_f(const Matrix& x, const Vector& y, const Vector& w,
                       Vector* gradient_f) {
    CHECK(gradient_f != nullptr);
    *gradient_f = x.transpose() * ((x * w) - y) + (l2_reg_ * w);
  }

  // L2 regularization parameter.
  double l2_reg_;

  // Cached number of iterations.
  int cached_iterations_;

  // Cached cost function value.
  double cached_cost_;
};

}  // namespace optimzer
}  // namespace ml

#endif  // ML_OPTIMIZER_GD_REGRESSOR_H_
