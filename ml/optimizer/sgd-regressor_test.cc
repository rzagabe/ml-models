// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/optimizer/sgd-regressor.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/squared-loss.h"
#include "ml/optimizer/ridge-loss.h"

using ml::optimizer::SGDRegressor;
using ml::optimizer::SquaredLoss;
using ml::optimizer::RidgeLoss;

TEST(SGDRegressor, SquaredLossOptimization) {
  Eigen::MatrixXf x(12, 1);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  Eigen::VectorXf y(12);
  y << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;

  Eigen::VectorXf w(1);
  w(0, 0) = 4;

  SGDRegressor optimizer;
  optimizer.Optimize(x, y, 0, 100, 0.0001, 0.001, &w);

  EXPECT_EQ(optimizer.cached_iterations(), 100);
}

TEST(SGDRegressor, RidgeLossOptimization) {
  Eigen::MatrixXf x(12, 1);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  Eigen::VectorXf y(12);
  y << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;

  Eigen::VectorXf w(1);
  w(0, 0) = 4;

  SGDRegressor optimizer;
  optimizer.set_l2_reg(0.0001);
  optimizer.Optimize(x, y, 0, 100, 0.0001, 0.001, &w);

  EXPECT_EQ(optimizer.cached_iterations(), 100);
}

TEST(SGDRegressor, SquaredLossMiniBatchOptimization) {
  Eigen::MatrixXf x(12, 1);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  Eigen::VectorXf y(12);
  y << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;

  Eigen::VectorXf w(1);
  w(0, 0) = 4;

  SGDRegressor optimizer;
  optimizer.set_batch_size(4);
  optimizer.Optimize(x, y, 0, 100, 0.0001, 0.001, &w);

  EXPECT_LT(optimizer.cached_iterations(), 60);
  EXPECT_LT(optimizer.cached_cost(), 0.0001);
}

TEST(SGDRegressor, BacktrackingLineSearchOptimization) {
  Eigen::MatrixXf x(12, 1);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  Eigen::VectorXf y(12);
  y << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;

  Eigen::VectorXf w(1);
  w(0, 0) = 4;

  SGDRegressor optimizer;
  optimizer.set_batch_size(4);
  optimizer.set_bt_line_search(true);
  optimizer.set_bt_line_search_alpha(0.3);
  optimizer.set_bt_line_search_beta(0.8);
  optimizer.Optimize(x, y, 0, 100, 0.0001, 0.001, &w);

  EXPECT_LT(optimizer.cached_iterations(), 30);
  EXPECT_LT(optimizer.cached_cost(), 0.0001);
}
