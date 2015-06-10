// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/stochastic-gradient-descent.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/optimizer.h"
#include "ml/optimizer/line-search.h"

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
    double f_k = 0.0;
    CHECK(loss_->eval_f(x, y, *w, &f_k))
        << "Couldn't compute objective function...";

    int index_instance = (batch_size_ * k) % x.rows();
    int num_instances = batch_size_;
    if (index_instance + num_instances > x.rows()) {
      num_instances =
          num_instances - (index_instance + num_instances - x.rows());
    }

    Vector grad_f_k;
    CHECK(loss_->eval_gradient_f(
        x.block(index_instance, 0, num_instances, x.cols()),
        y.block(index_instance, 0, num_instances, y.cols()), *w, &grad_f_k))
        << "Couldn't compute gradient...";

    cached_iterations_ = k + 1;
    cached_cost_ = f_k;

    LOG(INFO) << "iteration = " << k + 1 << ", cost = " << f_k;
    if (std::sqrt(grad_f_k.dot(grad_f_k)) < eps) {
      // TODO(zagabe.lu@gmail.com): Should the stopping criterion only
      // be tested against the minibatch used to compute the gradient?
      LOG(INFO) << "Stopping criterion attained (" << k + 1 << " iterations)";
      break;
    }

    if (bt_line_search_) {
      gamma = linesearch::BacktrackingLineSearch(loss_, x, y, *w, f_k, grad_f_k,
                                                 bt_line_search_alpha_,
                                                 bt_line_search_beta_);
    }

    *w = *w - gamma * grad_f_k;
    CHECK(loss_->intermediate_callback(x, y, k + 1, min_iter, max_iter, eps,
                                       &gamma, w));
  }

  return true;
}

}  // namespace optimizer
}  // namespace ml
