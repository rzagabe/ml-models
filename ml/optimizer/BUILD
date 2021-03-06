load("/tools/build_rules/proto_rules", "proto_library")

proto_library(
    name = "optimizer_proto",
    src = "optimizer.proto",
)

cc_library(
    name = "optimizers",
    deps = [
        ":optimizer",
        ":optimizer_proto",
        ":sgd-regressor",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "optimizer",
    srcs = ["optimizer.cc"],
    hdrs = ["optimizer.h"],
    deps = [
        ":optimizer_proto",
        "//ml:factory",
        "//third_party/eigen",
        "//third_party/glog",
    ],
)

cc_library(
    name = "loss-function",
    srcs = ["loss-function.cc"],
    hdrs = ["loss-function.h"],
    deps = [
        ":optimizer_proto",
        "//ml:factory",
        "//third_party/eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gd-regressor",
    srcs = ["gd-regressor.cc"],
    hdrs = ["gd-regressor.h"],
    deps = [
        ":optimizer",
        ":optimizer_proto",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "gd-regressor_test",
    srcs = ["gd-regressor_test.cc"],
    deps = [
        ":gd-regressor",
        "//third_party/glog",
        "//third_party/protobuf:proto_lib",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "sgd-regressor",
    srcs = ["sgd-regressor.cc"],
    hdrs = ["sgd-regressor.h"],
    deps = [
        ":optimizer",
        ":optimizer_proto",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "sgd-regressor_test",
    srcs = ["sgd-regressor_test.cc"],
    deps = [
        ":sgd-regressor",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "ridge-loss",
    srcs = ["ridge-loss.cc"],
    hdrs = ["ridge-loss.h"],
    deps = [
        ":loss-function",
        ":optimizer_proto",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "ridge-loss_test",
    srcs = ["ridge-loss_test.cc"],
    deps = [
        ":ridge-loss",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "squared-loss",
    srcs = ["squared-loss.cc"],
    hdrs = ["squared-loss.h"],
    deps = [
        ":loss-function",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "squared-loss_test",
    srcs = ["squared-loss_test.cc"],
    deps = [
        ":squared-loss",
        "//third_party/glog",
        "//third_party/eigen",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)
