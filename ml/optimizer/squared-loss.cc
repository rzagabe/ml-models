// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//

#include "ml/optimizer/squared-loss.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ml/optimizer/loss-function.h"

namespace ml {
namespace optimizer {

REGISTER_LOSS_FUNCTION(SquaredLoss)

}  // namespace optimizer
}  // namespace ml
