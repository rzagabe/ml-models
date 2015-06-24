// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/sgd-regressor.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace optimizer {

REGISTER_OPTIMIZER(SGDRegressor)

bool SGDRegressor::Optimize(const Matrix& x, const Vector& y, int min_iter,
                            int max_iter, double eps, double gamma, Vector* w) {
  CHECK(min_iter < max_iter) << "Unexpected iteration boundary";

  CHECK(batch_size_ > 0 && batch_size_ < x.rows())
      << "Unreasonable minibatch size: " << batch_size_;
  CHECK(w != nullptr);
  for (int k = 0; k < max_iter; ++k) {
    double f_k = eval_f(x, y, *w);

    int index_instance = (batch_size_ * k) % x.rows();
    int num_instances = batch_size_;
    if (index_instance + num_instances > x.rows()) {
      num_instances =
          num_instances - (index_instance + num_instances - x.rows());
    }

    Vector grad_f_k;
    eval_gradient_f(x.block(index_instance, 0, num_instances, x.cols()),
                    y.block(index_instance, 0, num_instances, y.cols()), *w,
                    &grad_f_k);

    cached_iterations_ = k + 1;
    cached_cost_ = f_k;

    LOG(INFO) << "iteration = " << k + 1 << ", cost = " << f_k;
    if (std::sqrt(grad_f_k.dot(grad_f_k)) < eps) {
      // TODO(zagabe.lu@gmail.com): Should the stopping criterion only
      // be tested against the minibatch used to compute the gradient?
      LOG(INFO) << "Stopping criterion attained (" << k + 1 << " iterations)";
      break;
    }

    // Backtracking line search.
    if (bt_line_search_) {
      gamma = 1.0;
      while (eval_f(x, y, *w - gamma * grad_f_k) >
             f_k - bt_line_search_alpha_ * gamma * grad_f_k.dot(grad_f_k))
        gamma = bt_line_search_beta_ * gamma;
      VLOG(1) << "step size: " << gamma;
    }

    *w = *w - gamma * grad_f_k;
  }

  return true;
}

}  // namespace optimizer
}  // namespace ml
