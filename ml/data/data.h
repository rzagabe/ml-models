// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// This is a simple implementation designed to facilitate the
// manipulation of the instances used to train the models.

#ifndef ML_DATA_DATA_H_
#define ML_DATA_DATA_H_

#include <iostream>
#include <memory>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace data {

using std::string;

class Data {
 public:
  typedef Eigen::MatrixXf Matrix;
  typedef Eigen::VectorXf Vector;

  friend class DataWriter;
  friend class DataReader;

  Data()
      : num_instances_(0),
        num_inputs_(0),
        targets_(kDefaultDataCapacity, 1),
        inputs_(kDefaultDataCapacity, kDefaultDataCapacity) {}

  virtual ~Data() {}

  int num_instances() const { return num_instances_; }

  int num_inputs() const { return num_inputs_; }

  Eigen::Block<const Eigen::VectorXf> targets() const {
    return targets_.block(0, 0, num_instances_, 1);
  }

  Eigen::Block<const Eigen::MatrixXf> inputs() const {
    return inputs_.block(0, 0, num_instances_, num_inputs_);
  }

 protected:
  friend class DataBuilder;

  const int kDefaultDataCapacity = 128;

  // Number of instances.
  int num_instances_;

  // Number of inputs per instances.
  int num_inputs_;

  // Column vector of targets.
  Vector targets_;

  // Matrix of inputs.
  Matrix inputs_;
};

}  // namespace data
}  // namespace ml

#endif  // ML_DATA_DATA_H_
