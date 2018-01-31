package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//...",
    ],
)

py_library(
    name = "accountant",
    srcs = [
        "accountant.py",
    ],
    deps = [
        "//dp_sgd/dp_optimizer:utils",
    ],
)

