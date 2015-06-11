#include "examples/hello-world-lib.h"

#include <iostream>
#include <string>

namespace hellolib {

const std::string HelloWorld::SayHello(const std::string& name) const {
  return "Hello, " + name + "!";
}

}  // namespace hellolib
