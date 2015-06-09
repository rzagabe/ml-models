// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#ifndef ML_DATA_LIBSVM_READER_H_
#define ML_DATA_LIBSVM_READER_H_

#include <iostream>
#include <memory>
#include <string>

#include "ml/data/data.h"
#include "ml/data/data-builder.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {
namespace data {

using std::shared_ptr;
using std::string;

class LibSVMReader {
 public:
  LibSVMReader(DataBuilder* builder) : builder_(builder) {}

  virtual ~LibSVMReader() {}

  Data* BuildFromIstream(std::istream* is);

  Data* BuildFromString(const string& data);

 private:
  // Data builder.
  shared_ptr<DataBuilder> builder_;
};

}  // namespace data
}  // namespace ml

#endif  // ML_DATA_LIBSVM_READER_H_
