load("/tools/build_rules/proto_rules", "proto_library")

proto_library(
    name = "optimizer_proto",
    src = "optimizer.proto",
)

cc_library(
    name = "optimizers",
    deps = [
        ":loss-function",
        ":squared-loss",
        ":ridge-loss",
        ":optimizer",
        ":optimizer_proto",
        ":gradient-descent",
        ":stochastic-gradient-descent",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "loss-function",
    srcs = ["loss-function.cc"],
    hdrs = ["loss-function.h"],
    deps = [
        "//ml:factory",
        "//external:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "optimizer",
    srcs = ["optimizer.cc"],
    hdrs = ["optimizer.h"],
    deps = [
        ":loss-function",
        ":optimizer_proto",
        "//ml:factory",
        "//external:glog",
        "//external:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gradient-descent",
    srcs = ["gradient-descent.cc"],
    hdrs = ["gradient-descent.h"],
    deps = [
        ":loss-function",
        ":optimizer",
        ":squared-loss",
        ":line-search",
        "//external:eigen",
        "//external:glog",
        "//external:gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "gradient-descent_test",
    srcs = ["gradient-descent_test.cc"],
    deps = [
        ":optimizer",
        ":optimizer_proto",
        ":gradient-descent",
        "//external:glog",
        "//external:proto_lib",
        "//external:gtest_main",
        "//ml:factory",
    ],
    size = "small",
)

cc_library(
    name = "stochastic-gradient-descent",
    srcs = ["stochastic-gradient-descent.cc"],
    hdrs = ["stochastic-gradient-descent.h"],
    deps = [
        ":squared-loss",
        ":ridge-loss",
        ":optimizer",
        ":line-search",
        "//external:eigen",
        "//external:glog",
        "//external:gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "stochastic-gradient-descent_test",
    srcs = ["stochastic-gradient-descent_test.cc"],
    deps = [
        ":stochastic-gradient-descent",
        "//external:eigen",
        "//external:glog",
        "//external:gtest_main",
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
        "//external:eigen",
        "//external:glog",
        "//external:gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "ridge-loss_test",
    srcs = ["ridge-loss_test.cc"],
    deps = [
        ":ridge-loss",
        "//external:eigen",
        "//external:glog",
        "//external:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "squared-loss",
    srcs = ["squared-loss.cc"],
    hdrs = ["squared-loss.h"],
    deps = [
        ":loss-function",
        "//external:eigen",
        "//external:glog",
        "//external:gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "squared-loss_test",
    srcs = ["squared-loss_test.cc"],
    deps = [
        ":squared-loss",
        "//external:glog",
        "//external:eigen",
        "//external:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "line-search",
    srcs = ["line-search.cc"],
    hdrs = ["line-search.h"],
    deps = [
        ":loss-function",
        "//external:eigen",
    ],
)
