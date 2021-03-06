// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/gd-regressor.h"

#include <iostream>
#include <string>
#include <memory>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

using ml::optimizer::GDRegressor;

TEST(GDRegressor, Optimize) {
  Eigen::MatrixXf x(12, 1);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  Eigen::VectorXf y(12);
  y << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;

  Eigen::VectorXf w(1);
  w << 1;

  GDRegressor optimizer;
  optimizer.Optimize(x, y, 0, 100, 0.0001, 0.001, &w);
  EXPECT_LT(optimizer.cached_iterations(), 20);
  EXPECT_LT(optimizer.cached_cost(), 0.0001);
}

TEST(GDRegressor, BacktrackingLineSearchOptimization) {
  Eigen::MatrixXf x(12, 1);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  Eigen::VectorXf y(12);
  y << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;

  Eigen::VectorXf w(1);
  w << 1;

  GDRegressor optimizer;
  optimizer.set_bt_line_search(true);
  optimizer.set_bt_line_search_alpha(0.3);
  optimizer.set_bt_line_search_beta(0.8);
  optimizer.Optimize(x, y, 0, 100, 0.0001, 0.001, &w);
  EXPECT_LT(optimizer.cached_iterations(), 20);
  EXPECT_LT(optimizer.cached_cost(), 0.0001);
}
