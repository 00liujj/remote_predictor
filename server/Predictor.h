#pragma once

#include "proto/predict.pb.h"
#include "proto/predict_service.grpc.pb.h"


typedef google::protobuf::Map<std::string, serving::TensorProto> ArgumentMap;


class Predictor {
public:
    virtual int init(const std::string& cfg, const std::string& weights) = 0;
    virtual int predict(const ArgumentMap& input, ArgumentMap* output) = 0;
    virtual int release() = 0;
};





