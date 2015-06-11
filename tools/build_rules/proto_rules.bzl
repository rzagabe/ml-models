# This is a dirty native macro to generate protobuf source files (cpp only).
# TODO(zagabe.lu@gmail.com):
#               - Clean this up...
#               - Add support for python and java.
proto_filetype = FileType([".proto"])

def proto_library(name, src, deps = [], generate_cc = True):
    includes = ""
    for v in deps:
        if v.find(':') != -1 and len(v[0:v.find(":")]) != 0:
            includes += "-I%s " % (v[2:v.find(":")])
    command = "$(location //third_party/protobuf:protoc) -I. %s --cpp_out=$(GENDIR) $(location %s)" % (includes, src)
    basename = src[0:-5]
    native.genrule(
        name = name + "_proto",
        srcs = [
            src,
            "//third_party/protobuf:protoc",
        ],
        cmd = command,
        outs = [
            basename + "pb.h",
            basename + "pb.cc",
        ],
    )
    native.cc_library(
        name = name,
        hdrs = [
            basename + "pb.h",
        ],
        srcs = [
            ":" + name + "_proto",
        ],
        deps = deps + [ "//third_party/protobuf:proto_lib" ],
    )
