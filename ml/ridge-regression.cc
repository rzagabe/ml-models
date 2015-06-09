// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/ridge-regression.h"

#include <fstream>
#include <iostream>
#include <string>

#include "Eigen/Core"
#include "Eigen/SVD"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "ml/data/data.h"
#include "ml/model.h"

namespace ml {

REGISTER_MODEL(RidgeRegression)

bool RidgeRegression::Initialize(const ::google::protobuf::Message& message) {
  const ModelParameters& parameters =
      dynamic_cast<const ModelParameters&>(message);
  return ParseFromProto(parameters);
}

bool RidgeRegression::Train(const data::Data& instances) {
  const Matrix& x = instances.inputs();
  Eigen::JacobiSVD<Matrix> svd;
  svd.compute(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Vector& y = instances.targets();
  const Matrix& s = svd.singularValues();
  auto const& nS = s.cwiseQuotient((s.array().square() + (l2_reg_ * l2_reg_))
                                       .matrix()).asDiagonal();
  w_ = svd.matrixV() * nS * svd.matrixU().transpose() * y;
  return true;
}

bool RidgeRegression::Update(const data::Data& instances) { return true; }

bool RidgeRegression::Predict(const data::Data& instances,
                              std::vector<double>* predictions) {
  const Matrix& x = instances.inputs();
  CHECK(x.cols() == w_.rows()) << "";
  Vector y = x * w_;
  for (int i = 0; i < y.rows(); ++i) {
    predictions->push_back(y(i, 0));
  }
  return true;
}

double RidgeRegression::Score(const data::Data& instances) {
  const Matrix& x = instances.inputs();
  const Vector& real_y = instances.targets();
  CHECK(x.cols() == w_.rows()) << "";
  Vector y = x * w_;
  return ((y - real_y).transpose() * (y - real_y)).sum() / y.rows();
}

bool RidgeRegression::SerializeToString(string* output) const {
  ModelParameters parameters;
  parameters.set_name(name_);
  parameters.set_type(type_);
  RidgeRegressionParameters* options =
      parameters.MutableExtension(RidgeRegressionParameters::options);
  options->set_l2_reg(l2_reg_);
  for (int i = 0; i < w_.rows(); ++i) {
    options->add_parameters(w_(i, 0));
  }
  return parameters.SerializeToString(output);
}

bool RidgeRegression::ParseFromFile(const string& filename) {
  std::fstream fs(filename, std::fstream::in);
  ModelParameters parameters;
  if (!parameters.ParseFromIstream(&fs)) return false;
  return ParseFromProto(parameters);
}

bool RidgeRegression::ParseFromString(const string& data) {
  ModelParameters parameters;
  if (!parameters.ParseFromString(data)) return false;
  return ParseFromProto(parameters);
}

bool RidgeRegression::ParseFromProto(const ModelParameters& parameters) {
  name_ = parameters.name();
  type_ = parameters.type();
  if (!parameters.HasExtension(RidgeRegressionParameters::options)) {
    LOG(ERROR) << "No RidgeRegressionParameters extension found";
    return false;
  }
  const RidgeRegressionParameters& options =
      parameters.GetExtension(RidgeRegressionParameters::options);
  l2_reg_ = options.l2_reg();
  w_.resize(options.parameters_size());
  for (int i = 0; i < options.parameters_size(); ++i) {
    w_(i, 0) = options.parameters(i);
  }
  return true;
}

}  // namespace ml
