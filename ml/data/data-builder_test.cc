// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/data-builder.h"

#include <iostream>
#include <string>
#include <memory>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"

const int kNumInstances = 3;

const int kNumFeatures = 4;

const char kDataMessageSample[] =
    "instances {"
    "   target: 1"
    "   inputs: 2"
    "   inputs: 3"
    "   inputs: 4"
    "}"
    "instances {"
    "   target: 1"
    "   inputs: 2"
    "   inputs: 3"
    "   inputs: 4"
    "}"
    "instances {"
    "   target: 1"
    "   inputs: 2"
    "   inputs: 3"
    "   inputs: 4"
    "}";

TEST(DataBuilder, Reset) {
  ml::data::DataBuilder builder;

  // Create two instances.
  builder.NewData();

  builder.NewInstance();
  builder.add_target(1.0);
  builder.add_input(5.0);
  builder.add_input(6.0);
  builder.add_input(7.0);

  // Reset data.
  builder.Reset();

  std::unique_ptr<ml::data::Data> instances(builder.GetData());

  // Check number of instances.
  EXPECT_EQ(0, instances->num_instances());
}

TEST(DataBuilder, AddTargets) {
  ml::data::DataBuilder builder;

  builder.NewData();

  builder.NewInstance();
  builder.add_target(1.0);
  builder.add_input(2.0);

  builder.NewInstance();
  builder.add_target(3.0);
  builder.add_input(4.0);

  std::unique_ptr<ml::data::Data> instances(builder.GetData());

  // Check number of instances.
  EXPECT_EQ(2, instances->num_instances());

  // // Check target column vector.
  EXPECT_EQ(2, instances->targets().rows());
  EXPECT_EQ(1, instances->targets().cols());

  EXPECT_EQ(2, instances->inputs().rows());
  EXPECT_EQ(1, instances->inputs().cols());
}

TEST(DataBuilder, AddInputs) {
  ml::data::DataBuilder builder;

  builder.NewData();

  builder.NewInstance();
  builder.add_target(1.0);
  builder.add_input(2.0);
  builder.add_input(3.0);
  builder.add_input(4.0);

  std::unique_ptr<ml::data::Data> instances(builder.GetData());

  // Check number of instances.
  EXPECT_EQ(1, instances->num_instances());

  Eigen::MatrixXf const& targets = instances->targets();
  EXPECT_EQ(1, targets.rows());

  Eigen::MatrixXf const& inputs = instances->inputs();
  EXPECT_EQ(3, inputs.cols());
}
