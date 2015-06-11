load("/tools/build_rules/proto_rules", "proto_library")

proto_library(
    name = "data_proto",
    src = "data.proto",
)

cc_library(
    name = "ml-data",
    deps = [
        ":data",
        ":data-builder",
        ":libsvm-reader",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "data",
    srcs = ["data.cc"],
    hdrs = ["data.h"],
    deps = [
        ":data_proto",
        "//third_party/protobuf:proto_lib",
        "//third_party/eigen",
        "//third_party/gtest",
        "//third_party/glog",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "data-writer",
    srcs = ["data-writer.cc"],
    hdrs = ["data-writer.h"],
    deps = [
        ":data",
        ":data_proto",
        "//third_party/eigen",
        "//third_party/gtest",
        "//third_party/glog",
    ],
)

cc_test(
    name = "data-writer_test",
    srcs = ["data-writer_test.cc"],
    deps = [
        ":data",
        ":data_proto",
        ":data-builder",
        ":data-writer",
        "//third_party/gtest:gtest_main",
        "//third_party/glog",
    ],
    size = "small",
)

cc_library(
    name = "data-reader",
    srcs = ["data-reader.cc"],
    hdrs = ["data-reader.h"],
    deps = [
        ":data",
        ":data_proto",
        "//third_party/eigen",
        "//third_party/gtest",
        "//third_party/glog",
    ],
)

cc_test(
    name = "data-reader_test",
    srcs = ["data-reader_test.cc"],
    deps = [
        ":data",
        ":data_proto",
        ":data-reader",
        "//third_party/protobuf:proto_lib",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "data-builder",
    srcs = ["data-builder.cc"],
    hdrs = ["data-builder.h"],
    deps = [
        ":data",
        "//third_party/gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "data-builder_test",
    srcs = ["data-builder_test.cc"],
    deps = [
        ":data",
        ":data-builder",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "libsvm-reader",
    srcs = ["libsvm-reader.cc"],
    hdrs = ["libsvm-reader.h"],
    deps = [
        ":data-builder",
        "//third_party/gtest",
        "//third_party/glog",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "libsvm-reader_test",
    srcs = ["libsvm-reader_test.cc"],
    deps = [
        ":data",
        ":data-builder",
        ":libsvm-reader",
        "//third_party/glog",
        "//third_party/eigen",
        "//third_party/gtest:gtest_main",
    ],
    data = [
        "//ml/data/testdata:simple.libsvm",
        "//ml/data/testdata:large.libsvm",
    ],
    size = "small",
)
