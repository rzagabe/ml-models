// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Provides an interface for regressor estimators.

#ifndef ML_OPTIMIZER_LOSS_FUNCTION_H_
#define ML_OPTIMIZER_LOSS_FUNCTION_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "ml/factory.h"

namespace ml {
namespace optimizer {

class LossFunction : public FactoryBase {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  LossFunction() {}

  virtual ~LossFunction() {}

  virtual bool eval_f(const Matrix& x, const Vector& y, const Vector& w,
                      double* f) = 0;

  virtual double eval_f(const Matrix& x, const Vector& y, const Vector& w) = 0;

  virtual bool eval_gradient_f(const Matrix& x, const Vector& y,
                               const Vector& w, Vector* gradient_f) = 0;

  virtual Vector eval_gradient_f(const Matrix& x, const Vector& y,
                                 const Vector& w) = 0;

  virtual bool intermediate_callback(const Matrix& x, const Vector& y,
                                     int cur_iter, int min_iter, int max_iter,
                                     double eps, double* gamma, Vector* w) = 0;
};

#define REGISTER_LOSS_FUNCTION(TYPE) REGISTER_CONCRETE(TYPE, TYPE, LossFunction)

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OPTIMIZER_LOSS_FUNCTION_H_
