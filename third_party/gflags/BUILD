licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gflags",
    srcs = [
        "repository/src/gflags.cc",
        "repository/src/gflags_reporting.cc",
        "repository/src/gflags_completions.cc",
    ],
    hdrs = [
        "build/include/gflags/config.h",
        "build/include/gflags/gflags_completions.h",
        "build/include/gflags/gflags_declare.h",
        "build/include/gflags/gflags_gflags.h",
        "build/include/gflags/gflags.h",
    ],
    copts = [
        "-Ithird_party/gflags/build/include/gflags",
    ],
    includes = [ 
        "build/include",
    ],
    linkopts = ["-lpthread"],
)
