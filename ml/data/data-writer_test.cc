// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/data-writer.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"
#include "ml/data/data.pb.h"
#include "ml/data/data-builder.h"

const int kNumInstances = 3;

const int kNumFeatures = 4;

const char kDataMessageSample[] =
    "instances {"
    "   target: 1"
    "   inputs: 2"
    "   inputs: 3"
    "   inputs: 4"
    "}"
    "instances {"
    "   target: 1"
    "   inputs: 2"
    "   inputs: 3"
    "   inputs: 4"
    "}"
    "instances {"
    "   target: 1"
    "   inputs: 2"
    "   inputs: 3"
    "   inputs: 4"
    "}";

TEST(DataWriter, Write) {
  ml::data::DataBuilder builder;

  // Create two instances.
  builder.NewData();
  
  builder.NewInstance();
  builder.add_target(1.0);
  builder.add_input(5.0);
  builder.add_input(6.0);
  builder.add_input(7.0);

  builder.NewInstance();
  builder.add_target(2.0);
  builder.add_input(8.0);
  builder.add_input(9.0);
  builder.add_input(10.0);

  std::unique_ptr<ml::data::Data> instances(builder.GetData());
  
  ml::data::DataMessage message;
  ml::data::DataWriter writer;
  writer.Write(*instances, &message);

  EXPECT_EQ(message.instances_size(), 2);
}
