#ifndef EXAMPLES_HELLO_WORLD_LIB_H_
#define EXAMPLES_HELLO_WORLD_LIB_H_

#include <iostream>
#include <string>

namespace hellolib {

class HelloWorld {
 public:
  HelloWorld() {}
  
  virtual ~HelloWorld() {}

  // Say hello to my friend!
  const std::string SayHello(const std::string& name) const;
};

}  // namespace hello

#endif  //  EXAMPLES_HELLO_WORLD_LIB_H_
