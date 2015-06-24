// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/gd-regressor.h"

#include <iostream>
#include <string>

#include "ml/optimizer/optimizer.h"

namespace ml {
namespace optimizer {

REGISTER_OPTIMIZER(GDRegressor)

bool GDRegressor::Optimize(const Matrix& x, const Vector& y, int min_iter,
                           int max_iter, double eps, double gamma, Vector* w) {
  CHECK(min_iter < max_iter) << "Unexpected iteration boundary";

  CHECK(w != nullptr);
  for (int k = 0; k < max_iter; ++k) {
    double f_k = eval_f(x, y, *w);

    Vector grad_f_k;
    eval_gradient_f(x, y, *w, &grad_f_k);

    LOG(INFO) << "iteration = " << k + 1 << ", cost = " << f_k;
    if (std::sqrt(grad_f_k.dot(grad_f_k)) < eps) {
      LOG(INFO) << "Stopping criterion attained (" << k + 1 << " iterations)";
      cached_iterations_ = k + 1;
      cached_cost_ = f_k;
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

}  // namespace optmizer
}  // namespace ml
