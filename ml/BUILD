load("/tools/build_rules/proto_rules", "proto_library")

cc_library(
    name = "factory",
    srcs = ["factory.cc"],
    hdrs = ["factory.h"],
    deps = [
        "//third_party/glog",
        "//third_party/protobuf:proto_lib",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "model_proto",
    src = "model.proto",
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    deps = [
        ":factory",
        ":model_proto",
        "//ml/data",
        "//third_party/protobuf:proto_lib",
        "//third_party/glog",
        "//third_party/gtest",
    ],
)

cc_library(
    name = "evaluator",
    srcs = ["evaluator.cc"],
    hdrs = ["evaluator.h"],
    deps = [
        ":model",
    ],
)

proto_library(
    name = "linear-regression_proto",
    src = "linear-regression.proto",
    deps = [
        ":model_proto",
        "//ml/optimizer:optimizers",
    ],
)

cc_library(
    name = "linear-regression",
    srcs = ["linear-regression.cc"],
    hdrs = ["linear-regression.h"],
    deps = [
        ":linear-regression_proto",
        ":model",
        ":model_proto",
        ":factory",
        "//ml/data",
        "//ml/optimizer:optimizers",
        "//third_party/eigen",
    ],
)

cc_test(
    name = "linear-regression_test",
    srcs = ["linear-regression_test.cc"],
    deps = [
        ":linear-regression",
        "//ml/data:ml-data",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
        "//third_party/protobuf:proto_lib",
    ],
    size = "small",
)

proto_library(
    name = "ridge-regression_proto",
    src = "ridge-regression.proto",
    deps = [
        ":model_proto",
    ],
)

cc_library(
    name = "ridge-regression",
    srcs = ["ridge-regression.cc"],
    hdrs = ["ridge-regression.h"],
    deps = [
        ":model",
        ":ridge-regression_proto",
        "//ml/data:ml-data",
        "//third_party/eigen",
        "//third_party/glog",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "ridge-regression_test",
    srcs = ["ridge-regression_test.cc"],
    deps = [
        ":ridge-regression",
        "//ml/data:ml-data",
        "//third_party/protobuf:proto_lib",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
    data = [
        "//ml/testdata:datasets/simple.libsvm",
    ],
)
