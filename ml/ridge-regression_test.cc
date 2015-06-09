// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/ridge-regression.h"

#include <fstream>
#include <iostream>
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "ml/data/data.h"
#include "ml/data/data-builder.h"
#include "ml/data/libsvm-reader.h"

using ml::RidgeRegression;
using ml::data::DataBuilder;
using ml::data::LibSVMReader;

const char kModelSpec[] =
    "name: \"RidgeRegression\" "
    "type: REGRESSION "
    "[ml.RidgeRegressionParameters.options]: {"
    "   l2_reg: 0.001 "
    " }";

const char kValidationSet[] =
    "400 1:1 2:200\n"
    "500 1:1 2:250\n"
    "1000 1:1 2:500\n"
    "5000 1:1 2:2500\n"
    "200 1:1 2:100\n"
    "250 1:1 2:125\n"
    "2000 1:1 2:1000\n"
    "20 1:1 2:10";

const char kTrainingSetPath[] = "ml/testdata/datasets/simple.libsvm";

class RidgeRegressionTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    LibSVMReader reader(new DataBuilder);
    std::fstream fs(kTrainingSetPath, std::fstream::in);
    training_.reset(reader.BuildFromIstream(&fs));
    validation_.reset(reader.BuildFromString(kValidationSet));
    CHECK(google::protobuf::TextFormat::ParseFromString(kModelSpec,
                                                        &model_parameters_));
  }

  std::unique_ptr<ml::data::Data> training_;
  std::unique_ptr<ml::data::Data> validation_;
  ml::ModelParameters model_parameters_;
};

TEST_F(RidgeRegressionTest, Train) {
  RidgeRegression model;
  CHECK(model.Initialize(model_parameters_));
  EXPECT_EQ(model.Train(*training_), true);
}

TEST_F(RidgeRegressionTest, Predict) {
  RidgeRegression model;
  CHECK(model.Initialize(model_parameters_));
  CHECK(model.Train(*training_));

  std::vector<double> predictions;
  EXPECT_EQ(model.Predict(*validation_, &predictions), true);
  EXPECT_EQ(predictions.size(), 8);
  const Eigen::VectorXf& y = validation_->targets();
  for (int i = 0; i < predictions.size(); ++i) {
    EXPECT_NEAR(predictions[i], y(i, 0), 0.05);
  }
}

TEST_F(RidgeRegressionTest, Score) {
  RidgeRegression model;
  CHECK(model.Initialize(model_parameters_));
  CHECK(model.Train(*training_));

  double score = model.Score(*validation_);
  EXPECT_NEAR(score, 0.0, 0.001);
}
