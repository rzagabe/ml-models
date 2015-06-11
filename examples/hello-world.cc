#include "examples/hello-world-lib.h"

#include <iostream>
#include <string>

int main() {
  hellolib::HelloWorld me;
  me.SayHello("Lucien");
  return 0;
}
