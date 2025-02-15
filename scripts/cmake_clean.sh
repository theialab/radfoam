#! /bin/sh

cmake-format -i CMakeLists.txt
cmake-format -i src/CMakeLists.txt
cmake-format -i external/CMakeLists.txt
cmake-format -i torch_bindings/CMakeLists.txt
