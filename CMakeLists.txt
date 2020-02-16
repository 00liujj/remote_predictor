cmake_minimum_required(VERSION 2.8)

set(CMAKE_TOOLCHAIN_FILE /home/jianjun/workspace/cross-compiling/toolchains/linux-gcc-4.9-x86_64-toolchain.cmake)

project(remote_predictor C CXX)


find_package(grpc)
find_package(MNN)
find_package(OpenCV)

set(CMAKE_BUILD_TYPE Debug)

add_definitions(-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0)

INCLUDE_DIRECTORIES(${grpc_INCLUDES} ${MNN_INCLUDES} ${OpenCV_INCLUDE_DIR} .)

file(GLOB PROTO_SRC_FILES
  proto/*.cc
)

file (GLOB CLIENT_SRC_FILES
  client/RemotePredictor.cpp
)

file (GLOB SERVER_SRC_FILES
  server/Predictor.cpp
  server/MNNPredictor.cpp
  server/PredictionServiceImpl.cpp
)

# client
add_library(client ${CLIENT_SRC_FILES} ${PROTO_SRC_FILES})
target_link_libraries(client ${grpc_LIBRARIES} -pthread)

add_executable(predict_client client/predict_client.cpp)
target_link_libraries(predict_client client ${OpenCV_LIBS})


## server
add_library(server ${SERVER_SRC_FILES} ${PROTO_SRC_FILES})
target_link_libraries(server ${grpc_LIBRARIES} ${MNN_LIBRARIES} -pthread)

add_executable(predict_server server/predict_server.cpp)
target_link_libraries(predict_server server)


## demo
add_executable(mnn_predict examples/mnn_predict.cpp)
target_link_libraries(mnn_predict ${MNN_LIBRARIES} ${OpenCV_LIBS})
