// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/libsvm-reader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <memory>

#include "Eigen/Core"
#include "ml/data/data.h"
#include "ml/data/data-builder.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

using ml::data::Data;
using ml::data::DataBuilder;

const int kSimpleNumInstances = 3;
const char kSimplePath[] = "ml/data/testdata/simple.libsvm";

TEST(LibSVMReader, SimpleBuild) {
  std::fstream fs(kSimplePath, std::fstream::in);
  ml::data::LibSVMReader reader(new DataBuilder);
  std::unique_ptr<Data> instances(reader.BuildFromIstream(&fs));
  EXPECT_EQ(kSimpleNumInstances, instances->num_instances());

  Eigen::MatrixXf real_inputs(3, 4);
  real_inputs << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  EXPECT_EQ(instances->inputs(), real_inputs);

  Eigen::VectorXf real_targets(3, 1);
  real_targets << 1, -1, 1;
  EXPECT_EQ(instances->targets(), real_targets);
}

const int kLargeNumInstances = 683;
const int kLargeNumFeatures = 10;
const char kLargePath[] = "ml/data/testdata/large.libsvm";

TEST(LibSVMReader, LargeBuild) {
  std::fstream fs(kLargePath, std::fstream::in);
  ml::data::LibSVMReader reader(new DataBuilder);
  std::unique_ptr<Data> instances(reader.BuildFromIstream(&fs));
  EXPECT_EQ(kLargeNumInstances, instances->num_instances());
  EXPECT_EQ(kLargeNumInstances, instances->targets().rows());
  EXPECT_EQ(kLargeNumFeatures, instances->inputs().cols());
}

const char kSimpleString[] =
    "1 1:0 4:9 2:3 3:8\n"
    "-1 1:0 3:10 2:3 4:7\n"
    "1 4:0 3:9 2:3 1:8";

TEST(LibSVMReader, SimpleStringBuild) {
  ml::data::LibSVMReader reader(new DataBuilder);
  std::unique_ptr<Data> instances(reader.BuildFromString(kSimpleString));
  EXPECT_EQ(instances->num_instances(), 3);
  EXPECT_EQ(instances->num_inputs(), 4);

  Eigen::MatrixXf real_inputs(3, 4);
  real_inputs << 0, 3, 8, 9, 0, 3, 10, 7, 8, 3, 9, 0;
  EXPECT_EQ(instances->inputs(), real_inputs);

  Eigen::VectorXf real_targets(3, 1);
  real_targets << 1, -1, 1;
  EXPECT_EQ(instances->targets(), real_targets);
}
