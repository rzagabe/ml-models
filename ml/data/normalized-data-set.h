#ifndef ML_DATA_NORMALIZED_DATA_H_
#define ML_DATA_NORMALIZED_DATA_H_

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace data {

class NormalizedData {
 public:
  NormalizedData() {}

  virtual ~NormalizedData() {}
}

}  // namespace data
}  // namespace ml

#endif  // ML_DATA_NORMALIZED_DATA_H_
