// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/data-builder.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace data {

void DataBuilder::NewInstance() {
  cur_input_index_ = 0;
  cur_instance_index_++;

  instances_->num_instances_ = cur_instance_index_ + 1;

  // Double targets matrix rows capacity if necessary.
  if (cur_instance_index_ >= instances_->targets_.rows()) {
    instances_->targets_.conservativeResize(instances_->targets_.rows() * 2,
                                            Eigen::NoChange_t());
  }

  // Double inputs matrix rows capacity if necessary.
  if (cur_instance_index_ >= instances_->inputs_.rows()) {
    instances_->inputs_.conservativeResize(instances_->inputs_.rows() * 2,
                                           Eigen::NoChange_t());
  }
}

void DataBuilder::Reset() {
  cur_input_index_ = 0;
  cur_instance_index_ = -1;
  instances_->num_instances_ = 0;
  instances_->num_inputs_ = 0;
}

void DataBuilder::add_target(float target) {
  CHECK(cur_instance_index_ >= 0)
      << "A new instance needs to be registered before adding more targets";
  instances_->targets_(cur_instance_index_, 0) = target;
}

void DataBuilder::add_raw_target(const std::string& raw_target) {
  float target;
  std::stringstream ss(raw_target);
  ss >> target;
  add_target(target);
}

void DataBuilder::add_input(float input) {
  CHECK(cur_instance_index_ >= 0)
      << "A new instance needs to be registered before adding more inputs";

  // Double input matrix cols capacity if necessary.
  if (cur_input_index_ >= instances_->inputs_.cols()) {
    instances_->inputs_.conservativeResize(Eigen::NoChange_t(),
                                           instances_->inputs_.cols() * 2);
  }

  instances_->inputs_(cur_instance_index_, cur_input_index_) = input;
  cur_input_index_++;
  if (cur_input_index_ > instances_->num_inputs_) instances_->num_inputs_++;
}

void DataBuilder::add_raw_input(const std::string& raw_input) {
  float input;
  std::stringstream ss(raw_input);
  ss >> input;
  add_input(input);
}

}  // namespace data
}  // namespace ml
