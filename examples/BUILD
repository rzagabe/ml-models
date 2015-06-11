cc_library(
    name = "hello-world-lib",
    srcs = ["hello-world-lib.cc"],
    hdrs = ["hello-world-lib.h"],
)

cc_test(
    name = "hello-world-lib_test",
    srcs = ["hello-world-lib_test.cc"],
    deps = [
        ":hello-world-lib",
        "//third_party/glog",
        "//third_party/gtest:gtest_main",
    ],
    size = "small",
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-world-lib",
    ],
)
