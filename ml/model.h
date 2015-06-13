// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)
//
// Unsophisticated model interface.

#ifndef ML_MODEL_H_
#define ML_MODEL_H_

#include "ml/model.pb.h"

#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "google/protobuf/message.h"
#include "ml/data/data.h"
#include "ml/factory.h"

namespace ml {

using std::string;

class Model : public FactoryBase {
 public:
  Model() : name_("") {}

  Model(const string& name) : name_(name), type_(ModelParameters::REGRESSION) {}
  
  virtual ~Model() {}

  // Initializes the model.
  virtual bool Initialize(const ModelParameters& parameters);

  // Fit the model with a given training set.
  virtual bool Train(const data::Data& instances) = 0;

  // Updates model states with new instances.
  virtual bool Update(const data::Data& instances) = 0;

  // Predicts target values of a given instances.
  virtual bool Predict(const data::Data& instances,
                       std::vector<double>* predictions) = 0;

  // Evaluates model's preformance against a given validation set.
  virtual double Score(const data::Data& instances) = 0;

  // Returns model's name.
  const string& name() const { return name_; }

  // Returns model's type.
  const ModelParameters::ModelType type() const { return type_; }

  // Saves the model state in a string.
  virtual bool SerializeToString(string* output) const;

  // Loads the model from a string.
  virtual bool ParseFromString(const string& data);

  // Loads the model from a file.
  virtual bool ParseFromFile(const string& filename);

  // Loads the model from model proto message.
  virtual bool ParseFromProto(const ModelParameters& parameters);

 protected:
  void set_name(string const& name) { name_ = name; }

  void set_type(ModelParameters::ModelType type) { type_ = type; }

  // Model's name.
  string name_;

  ModelParameters::ModelType type_;
};

#define REGISTER_MODEL(TYPE) REGISTER_CONCRETE(TYPE, TYPE, Model)

}  // namespace ml

#endif  // ML_MODEL_H_
