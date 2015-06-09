// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#ifndef ML_DATA_DATA_WRITER_H_
#define ML_DATA_DATA_WRITER_H_

#include "ml/data/data.pb.h"

#include <iostream>
#include <string>

#include "ml/data/data.h"

namespace ml {
namespace data {

class DataWriter {
 public:
  DataWriter() {}

  virtual ~DataWriter() {}
  
  virtual void Write(const Data& data, DataMessage* message) const;
};

}  // namespace data
}  // namespace ml

#endif  // ML_DATA_DATA_WRITER_H_
