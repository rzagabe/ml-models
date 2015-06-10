#ifndef ML_LINE_SEARCH_H_
#define ML_LINE_SEARCH_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {
namespace linesearch {

typedef Eigen::VectorXf Vector;
typedef Eigen::MatrixXf Matrix;

double BacktrackingLineSearch(LossFunction* loss, const Matrix& x,
                              const Vector& y, const Vector& w_k, double f_k,
                              const Vector& grad_f_k, double alpha,
                              double beta);

}  // namespace linesearch
}  // namespace optimizer
}  // namespace ml

#endif  // ML_LINE_SEARCH_H_
