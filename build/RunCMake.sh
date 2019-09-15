#!/bin/bash

build_type=${1:-Release}

# Need to specify like "60" or "75"
# Not to specify like "6.0" or "7.5"
compute_capability=${2}

echo "Build Type is [${build_type}]"
echo "Compute Capability is [${compute_capability}]"

cmake \
-D CMAKE_BUILD_TYPE=${build_type} \
-D CMAKE_CXX_COMPILER=/usr/local/clang_8.0.0/bin/clang++ \
-D CUDA_HOST_COMPILER=/usr/local/clang_8.0.0/bin/clang \
-D BUILD_UTILS=FALSE \
-D GLFW_BUILD_DOCS=FALSE \
-D GLFW_BUILD_EXAMPLES=FALSE \
-D GLFW_BUILD_TESTS=FALSE \
-D CUDA_TARGET_COMPUTE_CAPABILITY=${compute_capability} \
-L -G Ninja ..
