// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#ifndef ML_DATA_DATA_READER_H_
#define ML_DATA_DATA_READER_H_

#include "ml/data/data.pb.h"

#include <iostream>
#include <string>

#include "ml/data/data.h"

namespace ml {
namespace data {

class DataReader {
 public:
  DataReader() {}

  virtual ~DataReader() {}

  virtual void Read(const DataMessage& message, Data* instance);
};

}  // namespace data
}  // namespace ml

#endif  // ML_DATA_DATA_READER_H_
