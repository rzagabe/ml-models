licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gflags",
    srcs = [
        "repository/src/gflags.cc",
        "repository/src/gflags_reporting.cc",
        "repository/src/gflags_completions.cc",
    ],
    includes = [ 
        "repository/src",
        "repository/include",
        "repository/include/gflags",
    ],
    linkopts = ["-lpthread"],
)
