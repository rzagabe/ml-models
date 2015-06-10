// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/line-search.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {
namespace linesearch {

double BacktrackingLineSearch(LossFunction* loss, const Matrix& x,
                              const Vector& y, const Vector& w_k, double f_k,
                              const Vector& grad_f_k, double alpha,
                              double beta) {
  double t = 1.0;  // Step size.
  while (loss->eval_f(x, y, w_k - t * grad_f_k) >
         f_k - alpha * t * grad_f_k.dot(grad_f_k))
    t = beta * t;
  VLOG(1) << "step size: " << t;
  return t;
}

}  // namespace linesearch
}  // namespace optimzer
}  // namespace ml
