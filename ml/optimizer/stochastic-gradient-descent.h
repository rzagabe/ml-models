// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Provides a simple stochastic gradient descent optimization method.
// TODO(zagabe.lu@gmail.com):
//      - Randomize batches before computing gradient against.

#ifndef ML_OTPIMIZER_STOCHASTIC_GRADIENT_DESCENT_H_
#define ML_OTPIMIZER_STOCHASTIC_GRADIENT_DESCENT_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/loss-function.h"
#include "ml/optimizer/squared-loss.h"
#include "ml/optimizer/optimizer.h"

namespace ml {
namespace optimizer {

class StochasticGradientDescent : public Optimizer {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  StochasticGradientDescent()
      : batch_size_(1),
        cached_iterations_(0),
        cached_cost_(0.0),
        loss_(new SquaredLoss) {}

  virtual ~StochasticGradientDescent() { delete loss_; }

  virtual bool Initialize(const ::google::protobuf::Message& message) {
    const OptimizationParameters& options =
        dynamic_cast<const OptimizationParameters&>(message);
    const std::string& loss_name = options.loss_parameters().name();
    const LossParameters& loss_parameters = options.loss_parameters();
    delete loss_;
    loss_ = Factory<LossFunction>::CreateOrDie(loss_name, loss_parameters);
    batch_size_ = options.batch_size();
    bt_line_search_ = options.backtracking_line_search();
    bt_line_search_alpha_ = options.backtracking_line_search_alpha();
    bt_line_search_beta_ = options.backtracking_line_search_beta();
    return true;
  }

  // Executes the given optimization settings.
  virtual bool Optimize(const Matrix& x, const Vector& y, int min_iter,
                        int max_iter, double eps, double gamma, Vector* w);

  void set_batch_size(int batch_size) { batch_size_ = batch_size; }
  int batch_size() const { return batch_size_; }

  int cached_iterations() const { return cached_iterations_; }

  double cached_cost() const { return cached_cost_; }

  const Vector& cached_grad_f() const { return cached_grad_f_; }

  LossFunction* loss() const { return loss_; }
  void set_loss(LossFunction* loss) {
    CHECK(loss != nullptr);
    delete loss_;
    loss_ = loss;
  }

 private:
  // Batch size to compute the gradient against.
  int batch_size_;

  // Cached number of iterations.
  int cached_iterations_;

  // Cached cost function result.
  double cached_cost_;

  // Cached gradients.
  Vector cached_grad_f_;

  // Loss function to minimize.
  LossFunction* loss_;
};

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OTPIMIZER_STOCHASTIC_GRADIENT_DESCENT_H_
