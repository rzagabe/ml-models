// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/linear-regression.h"

#include <fstream>
#include <iostream>
#include <string>
#include <memory>

#include "Eigen/Core"
#include "glog/logging.h"
#include "ml/data/data.h"
#include "ml/factory.h"
#include "ml/optimizer/optimizer.h"

namespace ml {

using ml::Factory;
using ml::optimizer::Optimizer;

REGISTER_MODEL(LinearRegression);

bool LinearRegression::Initialize(const ::google::protobuf::Message& message) {
  const ModelParameters& parameters =
      dynamic_cast<const ModelParameters&>(message);
  return ParseFromProto(parameters);
}

bool LinearRegression::Train(const data::Data& instances) {
  LOG(INFO) << "Training the model...";
  const Matrix& x = instances.inputs();
  const Vector& y = instances.targets();

  if (w_.rows() != 0 && x.cols() > w_.rows()) {
    LOG(ERROR) << "Unexpected number of parameters.";
    return false;
  }

  if (w_.rows() == 0) {
    w_.resize(x.cols());
    w_.setZero();
  }

  LOG(INFO) << w_;
  const std::string& optimizer_name = optimization_parameters_.name();
  std::unique_ptr<Optimizer> optimizer(Factory<Optimizer>::CreateOrDie(
      optimizer_name, optimization_parameters_));
  if (!optimizer->Optimize(x, y, min_iter_, max_iter_, eps_, gamma_, &w_)) {
    LOG(ERROR) << "optimization failed";
    return false;
  }

  return true;
}

bool LinearRegression::Update(const data::Data& instances) { return true; }

bool LinearRegression::Predict(const data::Data& instances,
                               std::vector<double>* predictions) {
  const Matrix& x = instances.inputs();
  CHECK(x.cols() == w_.rows()) << "";
  Vector y = x * w_;
  for (int i = 0; i < y.rows(); ++i) {
    predictions->push_back(y(i, 0));
  }
  return true;
}

double LinearRegression::Score(const data::Data& instances) {
  const Matrix& x = instances.inputs();
  const Vector& real_y = instances.targets();
  CHECK(x.cols() == w_.rows()) << "";
  Vector y = x * w_;
  return ((y - real_y).transpose() * (y - real_y)).sum() / y.rows();
}

bool LinearRegression::SerializeToString(string* output) const {
  ModelParameters parameters;
  parameters.set_name(name_);
  parameters.set_type(type_);
  LinearRegressionParameters* options =
      parameters.MutableExtension(LinearRegressionParameters::options);
  options->set_max_iter(max_iter_);
  options->set_min_iter(min_iter_);
  options->set_eps(eps_);
  options->set_gamma(gamma_);
  (*options->mutable_optimization_parameters()) = optimization_parameters_;
  for (int i = 0; i < w_.rows(); ++i) {
    options->add_parameters(w_(i, 0));
  }
  return parameters.SerializeToString(output);
}

bool LinearRegression::ParseFromFile(const string& filename) {
  std::fstream fs(filename, std::fstream::in);
  ModelParameters parameters;
  if (!parameters.ParseFromIstream(&fs)) return false;
  return ParseFromProto(parameters);
}

bool LinearRegression::ParseFromString(const string& data) {
  ModelParameters parameters;
  if (!parameters.ParseFromString(data)) return false;
  return ParseFromProto(parameters);
}

bool LinearRegression::ParseFromProto(const ModelParameters& parameters) {
  name_ = parameters.name();
  type_ = parameters.type();
  if (!parameters.HasExtension(LinearRegressionParameters::options)) {
    LOG(ERROR) << "No LinearRegressionParameters extension";
    return false;
  }
  const LinearRegressionParameters& options =
      parameters.GetExtension(LinearRegressionParameters::options);
  max_iter_ = options.max_iter();
  min_iter_ = options.min_iter();
  gamma_ = options.gamma();
  optimization_parameters_ = options.optimization_parameters();
  w_.resize(options.parameters_size());
  for (int i = 0; i < options.parameters_size(); ++i) {
    w_(i, 0) = options.parameters(i);
  }
  return true;
}

}  // namespace ml
