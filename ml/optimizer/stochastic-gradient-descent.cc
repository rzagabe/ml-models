// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/stochastic-gradient-descent.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/optimizer.h"

namespace ml {
namespace optimizer {

// Register the modethod as specified in //ml/optimzer/optimizer.proto.
REGISTER_OPTIMIZER(StochasticGradientDescent)

bool StochasticGradientDescent::Optimize(const Matrix& x, const Vector& y,
                                         int min_iter, int max_iter, double eps,
                                         double gamma, Vector* w) {
  CHECK(loss_ != nullptr) << "No loss function initilized";
  CHECK(min_iter < max_iter) << "Unexpected iteration boundary";

  CHECK(batch_size_ > 0 && batch_size_ < x.rows())
      << "Unreasonable minibatch size: " << batch_size_;
  CHECK(w != nullptr);
  for (int k = 0; k < max_iter; ++k) {
    double cost = 0.0;
    CHECK(loss_->eval_f(x, y, *w, &cost))
        << "Couldn't compute objective function...";

    int index_instance = (batch_size_ * k) % x.rows();
    int num_instances = batch_size_;
    if (index_instance + num_instances > x.rows()) {
      num_instances =
          num_instances - (index_instance + num_instances - x.rows());
    }

    CHECK(loss_->eval_gradient_f(
        x.block(index_instance, 0, num_instances, x.cols()),
        y.block(index_instance, 0, num_instances, y.cols()), *w,
        &cached_grad_f_))
        << "Couldn't compute gradient...";

    cached_iterations_ = k + 1;
    cached_cost_ = cost;

    LOG(INFO) << "iteration = " << k + 1 << ", cost = " << cost;
    if (std::sqrt(cached_grad_f_.dot(cached_grad_f_)) < eps) {
      // TODO(zagabe.lu@gmail.com): Should the stopping criterion only
      // be tested against the minibatch used to compute the gradient?
      LOG(INFO) << "Stopping criterion attained (" << k + 1 << " iterations)";
      break;
    }

    // TODO(zagabe.lu@gmail.com): Execute a backtracking line search
    // algorithm.
    *w = *w - gamma * cached_grad_f_;
    CHECK(loss_->intermediate_callback(x, y, k + 1, min_iter, max_iter, eps,
                                       &gamma, w));
  }

  return true;
}

}  // namespace optimizer
}  // namespace ml
