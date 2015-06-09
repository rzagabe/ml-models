// Author: zagabe.lu@gmail.com (Lucien R. Zagabe)

#include "ml/data/data-reader.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "ml/data/data.h"
#include "ml/data/data.pb.h"

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

TEST(DataReader, Read) {
  ml::data::DataMessage message;
  CHECK(::google::protobuf::TextFormat::ParseFromString(kDataMessageSample,
                                                        &message));

  ml::data::Data instances;
  ml::data::DataReader reader;
  reader.Read(message, &instances);

  EXPECT_EQ(instances.num_instances(), kNumInstances);
}
