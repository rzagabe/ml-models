// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/model.h"

#include <fstream>
#include <iostream>
#include <string>

#include "google/protobuf/message.h"
#include "ml/factory.h"
#include "ml/model.pb.h"

namespace ml {

REGISTER_FACTORY(Model)

bool Model::Initialize(const ::google::protobuf::Message& message) {
  const ModelParameters& parameters =
      dynamic_cast<const ModelParameters&>(message);
  name_ = parameters.name();
  type_ = parameters.type();
  return true;
}

bool Model::SerializeToString(string* output) const {
  ModelParameters parameters;
  parameters.set_name(name_);
  parameters.set_type(type_);
  return parameters.SerializeToString(output);
}

bool Model::ParseFromString(const string& data) {
  ModelParameters parameters;
  if (!parameters.ParseFromString(data)) return false;
  return ParseFromProto(parameters);
}

bool Model::ParseFromFile(const string& filename) {
  std::fstream fs(filename, std::fstream::in);
  ModelParameters parameters;
  if (!parameters.ParseFromIstream(&fs)) return false;
  return ParseFromProto(parameters);
}

bool Model::ParseFromProto(const ModelParameters& parameters) {
  name_ = parameters.name();
  type_ = parameters.type();
  return true;
}

}  // namespace ml
