// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/libsvm-reader.h"

#include <queue>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <algorithm>

#include "ml/data/data.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace data {

using std::pair;
using std::vector;

Data* LibSVMReader::BuildFromIstream(std::istream* input) {
  CHECK(builder_->NewData()) << "Couldn't create new data";

  string line;
  while (std::getline(*input, line)) {
    vector<string> tokens;
    std::istringstream iss(line);
    copy(std::istream_iterator<string>(iss), std::istream_iterator<string>(),
         std::back_inserter(tokens));

    builder_->NewInstance();
    builder_->add_raw_target(tokens[0]);  // Add new target.

    std::priority_queue<pair<int, float>, vector<pair<int, float> >,
                        std::greater<pair<int, float> > > features;
    for (int i = 1; i < tokens.size(); ++i) {
      string token = tokens[i];
      std::stringstream ss;
      std::replace(token.begin(), token.end(), ':', ' ');
      int feature_index;
      float feature_value;
      ss << token;
      ss >> feature_index >> feature_value;
      features.push(std::make_pair(feature_index, feature_value));
    }

    // Add all features in non-decreasing order based on their index.
    while (!features.empty()) {
      builder_->add_input(features.top().second);
      features.pop();
    }
    VLOG(1) << "New instance added (current: "
            << builder_->GetData()->num_instances()
            << ", num_inputs: " << builder_->GetData()->num_inputs() << ")";
  }

  return builder_->GetData();
}

Data* LibSVMReader::BuildFromString(const std::string& data) {
  std::istringstream iss(data);
  return BuildFromIstream(&iss);
}

}  // namespace data
}  // namespace ml
