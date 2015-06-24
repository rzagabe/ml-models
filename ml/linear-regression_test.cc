// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/linear-regression.h"

#include <iostream>
#include <string>
#include <memory>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"
#include "ml/data/data-builder.h"
#include "ml/data/libsvm-reader.h"

using ml::LinearRegression;
using ml::data::DataBuilder;
using ml::data::LibSVMReader;

const char kLinearRegressionSpec[] =
    "name: \"LinearRegression\" "
    "type: REGRESSION "
    "[ml.LinearRegressionParameters.options]: { "
    "   max_iter: 100 "
    "   min_iter: 0 "
    "   eps: 0.0001 "
    "   gamma: 0.001 "
    "   optimizer_parameters: { "
    "           method: GRADIENT "
    "           batch_size: 1 "
    "           l2_regularization: 0.0 "
    "   } "
    "}";

const char kLibSVMTrainingSet[] =
    "2 1:1 2:1\n"
    "4 1:1 2:2\n"
    "6 1:1 2:3\n"
    "8 1:1 2:4\n"
    "10 1:1 2:5\n"
    "12 1:1 2:6\n"
    "20 1:1 2:10\n"
    "40 1:1 2:20\n";

const char kLibSVMInputSet[] =
    "0 1:1 2:16\n"
    "0 1:1 2:18\n"
    "0 1:1 2:22\n"
    "0 1:1 2:30\n";

const char kLibSVMValidationSet[] =
    "100 1:1 2:50\n"
    "120 1:1 2:60\n"
    "140 1:1 2:70\n"
    "160 1:1 2:80\n";

class LinearRegresssionTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    LibSVMReader reader(new DataBuilder);
    training_.reset(reader.BuildFromString(kLibSVMTrainingSet));
    input_.reset(reader.BuildFromString(kLibSVMInputSet));
    validation_.reset(reader.BuildFromString(kLibSVMValidationSet));
    CHECK(google::protobuf::TextFormat::ParseFromString(kLinearRegressionSpec,
                                                        &model_parameters_));
  }

  std::unique_ptr<ml::data::Data> input_;
  std::unique_ptr<ml::data::Data> training_;
  std::unique_ptr<ml::data::Data> validation_;
  ml::ModelParameters model_parameters_;
};

TEST_F(LinearRegresssionTest, Train) {
  LinearRegression model;
  CHECK(model.Initialize(model_parameters_));
  EXPECT_EQ(model.Train(*training_), true);
}

TEST_F(LinearRegresssionTest, Predict) {
  LinearRegression model;
  CHECK(model.Initialize(model_parameters_));
  CHECK(model.Train(*training_));

  std::vector<double> predictions;
  EXPECT_EQ(model.Predict(*input_, &predictions), true);
  EXPECT_EQ(predictions.size(), 4);
  const Eigen::MatrixXf& x = input_->inputs();
  LOG(INFO) << x;
  for (int i = 0; i < predictions.size(); ++i) {
    EXPECT_NEAR(predictions[i], x(i, 1) * 2, 0.5);
  }
}

TEST_F(LinearRegresssionTest, Score) {
  LinearRegression model;
  CHECK(model.Initialize(model_parameters_));
  CHECK(model.Train(*training_));

  double score = model.Score(*validation_);
  EXPECT_NEAR(score, 0.5, 0.2);
}
