set -ex

CXX=g++

source $HOME/workspace/cross-compiling/thirdparty/linux-gcc-4.9-x86_64/grpc-1.26.0/grpc-vars.sh

protoc --cpp_out=. *.proto 
protoc --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` *_service.proto 

