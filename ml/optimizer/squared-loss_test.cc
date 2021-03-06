// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#include "ml/optimizer/squared-loss.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(SquaredLoss, EvalF) {
  Eigen::MatrixXf x(6, 1);
  x << 1, 2, 3, 4, 5, 6;

  Eigen::VectorXf y(6);
  y << 2, 4, 6, 8, 10, 12;

  Eigen::VectorXf w(1);
  w(0, 0) = 2;

  ml::optimizer::SquaredLoss loss;
  double rss;
  CHECK(loss.eval_f(x, y, w, &rss));

  // Check cost function.
  EXPECT_EQ(rss, 0.0);
}

TEST(SquaredLoss, EvalGradientF) {
  Eigen::MatrixXf x(6, 1);
  x << 1, 2, 3, 4, 5, 6;

  Eigen::VectorXf y(6);
  y << 2, 4, 6, 8, 10, 12;

  Eigen::VectorXf w(1);
  w(0, 0) = 3;

  ml::optimizer::SquaredLoss loss;

  Eigen::VectorXf gradient(1);
  CHECK(loss.eval_gradient_f(x, y, w, &gradient));

  EXPECT_GT(gradient.sum(), 0);
}
