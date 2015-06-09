// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/data-reader.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"

namespace ml {
namespace data {

void DataReader::Read(const DataMessage& message, Data* instances) {
  for (int i = 0; i < message.instances_size(); ++i) {
    instances->num_instances_++;
    instances->num_inputs_ = message.instances(i).inputs_size();

    // Resize matrix if necessary.
    if (instances->num_instances_ > instances->targets_.rows()) {
      instances->targets_.conservativeResize(instances->targets_.rows() * 2,
                                             Eigen::NoChange_t());
      instances->inputs_.conservativeResize(instances->inputs_.rows() * 2,
                                            Eigen::NoChange_t());
    }
    instances->targets_(instances->num_instances_ - 1, 0) =
        message.instances(i).target();

    // Resize matrix if necessary.
    if (instances->num_inputs_ > instances->inputs_.cols()) {
      instances->inputs_.conservativeResize(Eigen::NoChange_t(),
                                            instances->inputs_.cols() * 2);
    }

    for (int j = 0; j < instances->num_inputs_; ++j) {
      instances->inputs_(instances->num_instances_ - 1,
                         instances->num_inputs_ - 1) =
          message.instances(i).inputs(j);
    }
  }
}

}  // namespace data
}  // namespace ml
