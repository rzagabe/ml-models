// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Gradient Descent implementation.

#ifndef ML_OPTIMIZER_BATCH_GRADIENT_DESCENT_REGRESSOR_H_
#define ML_OPTIMIZER_BATCH_GRADIENT_DESCENT_REGRESSOR_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/optimizer.h"
#include "ml/optimizer/loss-function.h"
#include "ml/optimizer/squared-loss.h"

namespace ml {
namespace optimizer {

class GradientDescent : public Optimizer {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  GradientDescent()
      : cached_iterations_(0), cached_cost_(0.0), loss_(new SquaredLoss) {}

  virtual ~GradientDescent() { delete loss_; }

  virtual bool Initialize(const OptimizationParameters& parameters) {
    const std::string& loss_name = parameters.loss_parameters().name();
    const LossParameters& loss_parameters = parameters.loss_parameters();
    delete loss_;
    loss_ = Factory<LossFunction>::CreateOrDie(loss_name);
    if (!loss_->Initialize(loss_parameters)) return false;
    bt_line_search_ = parameters.bt_line_search();
    bt_line_search_alpha_ = parameters.bt_line_search_alpha();
    bt_line_search_beta_ = parameters.bt_line_search_beta();
    return true;
  }

  // Executes the given optimization settings.
  virtual bool Optimize(const Matrix& x, const Vector& y, int min_iter,
                        int max_iter, double eps, double gamma, Vector* w);

  int cached_iterations() const { return cached_iterations_; }

  double cached_cost() const { return cached_cost_; }

  LossFunction* loss() const { return loss_; }
  void set_loss(LossFunction* loss) {
    CHECK(loss != nullptr);
    delete loss_;
    loss_ = loss;
  }

 private:
  // Cached number of iterations.
  int cached_iterations_;

  // Cached cost function value.
  double cached_cost_;

  // Loss function to minimize.
  LossFunction* loss_;
};

}  // namespace optimzer
}  // namespace ml

#endif  // ML_OPTIMIZER_BATCH_GRADIENT_DESCENT_REGRESSOR_H_
