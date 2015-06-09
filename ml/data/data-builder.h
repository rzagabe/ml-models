// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#ifndef ML_DATA_DATA_BUILDER_H_
#define ML_DATA_DATA_BUILDER_H_

#include "ml/data/data.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace data {

using std::string;

class DataBuilder {
 public:
  DataBuilder()
      : cur_input_index_(0), cur_instance_index_(-1), instances_(nullptr) {}

  virtual ~DataBuilder() {}

  bool NewData() {
    instances_ = new Data;
    cur_input_index_ = 0;
    cur_instance_index_ = -1;
    return true;
  }

  Data* GetData() { return instances_; }

  virtual void NewInstance();

  virtual void Reset();

  virtual void add_target(float target);

  virtual void add_raw_target(string const& raw_target);

  virtual void add_input(float input);

  virtual void add_raw_input(string const& raw_target);

 protected:
  // Current instance index.
  int cur_instance_index_;

  // Current feature index.
  int cur_input_index_;

  // Instances.
  Data* instances_;
};

}  // namespace data
}  // namespace ml

#endif  // ML_DATA_DATA_BUILDER_H_
