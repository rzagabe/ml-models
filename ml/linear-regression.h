// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Linear regression model implementation.

#ifndef ML_LINEAR_REGRESSION_H_
#define ML_LINEAR_REGRESSION_H_

#include "ml/linear-regression.pb.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"
#include "ml/model.h"
#include "ml/model.pb.h"

namespace ml {

class LinearRegression : public Model {
 public:
  typedef Eigen::VectorXf Vector;
  typedef Eigen::MatrixXf Matrix;

  LinearRegression()
      : Model("LinearRegression"),
        w_(0, 1),
        max_iter_(100),
        min_iter_(0),
        eps_(0.0001),
        gamma_(0.001) {}

  LinearRegression(const string& name)
      : Model(name),
        w_(0, 1),
        max_iter_(100),
        min_iter_(0),
        eps_(0.0001),
        gamma_(0.001) {}

  virtual ~LinearRegression() {}

  // Initialize linear regression model.
  virtual bool Initialize(const ::google::protobuf::Message& message);

  // Trains the model by minimizing the optimizer objective function.
  virtual bool Train(const data::Data& instances);

  // Updates the model parameters.
  virtual bool Update(const data::Data& instances);

  // Predicts target values of the given input set.
  virtual bool Predict(const data::Data& instances,
                       std::vector<double>* predictions);

  // Evaluates the model against a given validation set. (i.e. mean
  // squared error)
  virtual double Score(const data::Data& instances);

  // Saves the model current state.
  virtual bool SerializeToString(string* output) const;

  // Loads the model from a string.
  virtual bool ParseFromString(const string& data);

  // Loads the model from a file.
  virtual bool ParseFromFile(const string& filename);

  // Loads the model from a proto message.
  virtual bool ParseFromProto(const ModelParameters& parameters);

  int max_iter() { return max_iter_; }
  void set_max_iter(int max_iter) { max_iter_ = max_iter; }

  int min_iter() { return min_iter_; }
  void set_min_iter(int min_iter) { min_iter_ = min_iter; }

  int eps() { return eps_; }
  void set_eps(int eps) { eps_ = eps; }

  int gamma() { return gamma_; }
  void set_gamma(int gamma) { gamma_ = gamma; }

 private:
  // Regression coefficients.
  Vector w_;

  // Maximum optimization iterations.
  int max_iter_;

  // Minimum optimization iterations.
  int min_iter_;

  // Stopping criterions.
  double eps_;

  // Learning rate.
  double gamma_;

  // Optimization settings.
  optimizer::OptimizationParameters optimization_parameters_;
};

}  // namespace ml

#endif  // ML_LINEAR_REGRESSION_H_
