// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#ifndef ML_FACTORY_H_
#define ML_FACTORY_H_

#include <iostream>
#include <string>
#include <unordered_map>

#include "google/protobuf/message.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ml {

template <typename T>
class Creator {
 public:
  virtual T* New() const = 0;
};

class FactoryBase {
 public:
  virtual ~FactoryBase() {}
};

template <typename T>
class Factory {
 public:
  static const Creator<T>* Register(const std::string& name,
                                    Creator<T>* creator) {
    if (!initialized_) {
      creators_ = new std::unordered_map<std::string, const Creator<T>*>;
      initialized_ = 1;
    }
    (*creators_)[name] = creator;
    return creator;
  }

  static T* CreateOrDie(const std::string& name) {
    CHECK(creators_->find(name) != creators_->end()) << "Creator not found";
    T* instance = (*creators_)[name]->New();
    CHECK(instance != nullptr);
    return instance;
  }

  static bool IsRegistred(const std::string& name) {
    return creators_->find(name) != creators_->end();
  }

 private:
  static int initialized_;
  static std::unordered_map<std::string, const Creator<T>*>* creators_;
};

#define REGISTER_CONCRETE(TYPE, NAME, BASE)          \
  class TYPE##Creator : public Creator<BASE> {       \
   public:                                           \
    virtual BASE* New() const { return new TYPE(); } \
  };                                                 \
  const Creator<BASE>* TYPE##_creator =              \
      Factory<BASE>::Register(std::string(#NAME), new TYPE##Creator());

#define REGISTER_FACTORY(BASE)                           \
  template <>                                            \
  int Factory<BASE>::initialized_ = 0;                   \
  template <>                                            \
  std::unordered_map<std::string, const Creator<BASE>*>* \
      Factory<BASE>::creators_ = 0;

}  // namespace ml

#endif  // ML_FACTORY_H_
