#include "examples/hello-world-lib.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(HelloWorld, SayHello) {
  hellolib::HelloWorld me;
  EXPECT_EQ(me.SayHello("Lucien"), "Hello, Lucien!");
}
