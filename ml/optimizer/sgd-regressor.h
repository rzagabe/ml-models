// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Provides a simple stochastic gradient descent optimization method.
// TODO(zagabe.lu@gmail.com):
//      - Randomize batches before computing gradient against.

#ifndef ML_OTPIMIZER_SGD_REGRESSOR_H_
#define ML_OTPIMIZER_SGD_REGRESSOR_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/optimizer.h"
#include "ml/optimizer/optimizer.pb.h"

namespace ml {
namespace optimizer {

class SGDRegressor : public Optimizer {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  SGDRegressor()
      : batch_size_(1),
        l2_reg_(0.0),
        cached_iterations_(0),
        cached_cost_(0.0) {}

  virtual ~SGDRegressor() {}

  // Initialize optimizer.
  virtual bool Initialize(const OptimizerParameters& parameters) {
    batch_size_ = parameters.batch_size();
    l2_reg_ = parameters.l2_regularization();
    bt_line_search_ = parameters.bt_line_search();
    bt_line_search_alpha_ = parameters.bt_line_search_alpha();
    bt_line_search_beta_ = parameters.bt_line_search_beta();
    return true;
  }

  // Executes the given optimization settings.
  virtual bool Optimize(const Matrix& x, const Vector& y, int min_iter,
                        int max_iter, double eps, double gamma, Vector* w);

  void set_batch_size(int batch_size) { batch_size_ = batch_size; }
  int batch_size() const { return batch_size_; }

  void set_l2_reg(double l2_reg) { l2_reg_ = l2_reg; }
  double l2_reg() const { return l2_reg_; }

  int cached_iterations() const { return cached_iterations_; }
  double cached_cost() const { return cached_cost_; }
  const Vector& cached_grad_f() const { return cached_grad_f_; }

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

  // Batch size to compute the gradient against.
  int batch_size_;

  // L2 regularization parameter.
  double l2_reg_;

  // Cached number of iterations.
  int cached_iterations_;

  // Cached cost function result.
  double cached_cost_;

  // Cached gradients.
  Vector cached_grad_f_;
};

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OTPIMIZER_SGD_REGRESSOR_H_
