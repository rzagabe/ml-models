// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Trivial ridge regression model implementation via SVD.

#ifndef ML_RIDGE_REGRESSION_H_
#define ML_RIDGE_REGRESSION_H_

#include "ml/ridge-regression.pb.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "ml/data/data.h"
#include "ml/model.h"

namespace ml {

using std::string;

class RidgeRegression : public Model {
 public:
  typedef Eigen::MatrixXf Matrix;
  typedef Eigen::VectorXf Vector;

  RidgeRegression() : Model("RidgeRegression"), l2_reg_(0) {}

  RidgeRegression(const string& name) : Model(name), l2_reg_(0) {}

  virtual ~RidgeRegression() {}

  // Initiliaze ridge regression model.
  virtual bool Initialize(const ModelParameters& parameters);

  // Trains the model by using SVD to approximate the coeficients.
  virtual bool Train(const data::Data& instances);

  // Updates the model parameters.
  virtual bool Update(const data::Data& instances);

  // Predict target values of the given input set.
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

  // Loads the model from a proto.
  virtual bool ParseFromProto(const ModelParameters& parameters);

 private:
  // Regression coeficients.
  Vector w_;

  // L2 regularization.
  double l2_reg_;
};

}  // namespace ml

#endif  // ML_RIDGE_REGRESSION_H_
