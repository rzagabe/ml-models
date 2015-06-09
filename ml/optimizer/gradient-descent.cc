// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#include "ml/optimizer/gradient-descent.h"

#include <iostream>
#include <string>

#include "ml/optimizer/optimizer.h"

namespace ml {
namespace optimizer {

// Register the modethod as specified in //ml/optimzer/optimizer.proto.
REGISTER_OPTIMIZER(GradientDescent)

bool GradientDescent::Optimize(const Matrix& x, const Vector& y, int min_iter,
                               int max_iter, double eps, double gamma,
                               Vector* w) {
  CHECK(loss_ != nullptr) << "No loss function initilized";
  CHECK(min_iter < max_iter) << "Unexpected iteration boundary";

  CHECK(w != nullptr);
  for (int k = 0; k < max_iter; ++k) {
    double cost;
    CHECK(loss_->eval_f(x, y, *w, &cost));

    Vector grad_f;
    CHECK(loss_->eval_gradient_f(x, y, *w, &grad_f));

    LOG(INFO) << "iteration = " << k + 1 << ", cost = " << cost;
    if (std::abs(cost) < eps) {
      LOG(INFO) << "Stopping criterion attained (" << k + 1 << " iterations)";
      cached_iterations_ = k + 1;
      cached_cost_ = cost;
      break;
    }

    *w = *w - gamma * grad_f;
    CHECK(loss_->intermediate_callback(x, y, k + 1, min_iter, max_iter, eps,
                                       &gamma, w));
  }

  return true;
}

}  // namespace optmizer
}  // namespace ml
