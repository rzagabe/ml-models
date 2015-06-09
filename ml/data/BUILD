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
        "//external:proto_lib",
        "//external:eigen",
        "//external:gtest",
        "//external:glog",
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
        "//external:eigen",
        "//external:gtest",
        "//external:glog",
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
        "//external:gtest_main",
        "//external:glog",
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
        "//external:eigen",
        "//external:gtest",
        "//external:glog",
    ],
)

cc_test(
    name = "data-reader_test",
    srcs = ["data-reader_test.cc"],
    deps = [
        ":data",
        ":data_proto",
        ":data-reader",
        "//external:proto_lib",
        "//external:glog",
        "//external:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "data-builder",
    srcs = ["data-builder.cc"],
    hdrs = ["data-builder.h"],
    deps = [
        ":data",
        "//external:gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "data-builder_test",
    srcs = ["data-builder_test.cc"],
    deps = [
        ":data",
        ":data-builder",
        "//external:glog",
        "//external:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "libsvm-reader",
    srcs = ["libsvm-reader.cc"],
    hdrs = ["libsvm-reader.h"],
    deps = [
        ":data-builder",
        "//external:gtest",
        "//external:glog",
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
        "//external:glog",
        "//external:eigen",
        "//external:gtest_main",
    ],
    data = [
        "//ml/data/testdata:simple.libsvm",
        "//ml/data/testdata:large.libsvm",
    ],
    size = "small",
)
