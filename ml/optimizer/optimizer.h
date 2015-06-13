// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#ifndef ML_OPTIMIZER_OPTIMIZER_H_
#define ML_OPTIMIZER_OPTIMIZER_H_

#include "ml/optimizer/optimizer.pb.h"

#include <iostream>
#include <string>
#include <memory>

#include "Eigen/Core"
#include "glog/logging.h"
#include "ml/factory.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {

class Optimizer : public FactoryBase {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  Optimizer()
      : bt_line_search_(false),
        bt_line_search_alpha_(0.3),
        bt_line_search_beta_(0.8) {}

  virtual ~Optimizer() {}

  virtual bool Initialize(const OptimizationParameters& parameters) {
    bt_line_search_ = parameters.bt_line_search();
    bt_line_search_alpha_ = parameters.bt_line_search_alpha();
    bt_line_search_beta_ = parameters.bt_line_search_beta();
  }

  virtual bool Optimize(const Matrix& x, const Vector& y, int min_iter,
                        int max_iter, double eps, double gamma, Vector* w) = 0;

  bool bt_line_search() { return bt_line_search_; }
  void set_bt_line_search(bool bt_line_search) {
    bt_line_search_ = bt_line_search;
  }

  bool bt_line_search_alpha() { return bt_line_search_alpha_; }
  void set_bt_line_search_alpha(double bt_line_search_alpha) {
    bt_line_search_alpha_ = bt_line_search_alpha;
  }

  bool bt_line_search_beta() { return bt_line_search_beta_; }
  void set_bt_line_search_beta(double bt_line_search_beta) {
    bt_line_search_beta_ = bt_line_search_beta;
  }

 protected:
  // If true, choose step size via backtracking line search.
  bool bt_line_search_;

  // Backtracking line search constant. (0 <= alpha <= 0.5)
  double bt_line_search_alpha_;

  // Backtracking line search constant. (0 <= beta <= 1)
  double bt_line_search_beta_;
};

#define REGISTER_OPTIMIZER(TYPE) REGISTER_CONCRETE(TYPE, TYPE, Optimizer)

}  // namespace optimizer
}  // namespace ml

#endif  // ML_OPTIMIZER_OPTIMIZER_H_
