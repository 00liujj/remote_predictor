cmake_minimum_required(VERSION 2.8)

set(CMAKE_TOOLCHAIN_FILE /home/jianjun/workspace/cross-compiling/toolchains/linux-gcc-4.9-x86_64-toolchain.cmake)

project(remote_detector C CXX)


find_package(grpc)

find_package(MNN)


add_definitions(-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0)

INCLUDE_DIRECTORIES(${grpc_INCLUDES} ${MNN_INCLUDES} .)

file(GLOB SRC_FILES
  proto/*.cc
)


add_library(remote_predictor SHARED client/RemotePredictor.cpp ${SRC_FILES})
target_link_libraries(remote_predictor ${grpc_LIBRARIES} -pthread)
set_target_properties(remote_predictor
    PROPERTIES LINK_FLAGS "-Wl,-z,defs"
)


add_executable(predict_server server/Predictor.cpp ${SRC_FILES})
target_link_libraries(predict_server ${grpc_LIBRARIES} -pthread)