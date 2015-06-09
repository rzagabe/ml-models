// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#include "ml/optimizer/ridge-loss.h"

#include <iostream>
#include <string>

#include "Eigen/Core"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {

REGISTER_LOSS_FUNCTION(RidgeLoss)

}  // namespace optimizer
}  // namespace ml
