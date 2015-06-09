// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/data-writer.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"

namespace ml {
namespace data {

void DataWriter::Write(const Data& instances, DataMessage* message) const {
  for (int i = 0; i < instances.num_instances_; ++i) {
    Instance* instance = message->add_instances();
    instance->set_target(instances.targets_(i, 0));
    for (int j = 0; j < instances.num_inputs_; ++j) {
      instance->add_inputs(instances.inputs_(i, j));
    }
  }
}

}  // namespace data
}  // namespace ml
