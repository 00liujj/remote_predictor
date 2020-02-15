#pragma once

#include "proto/predict.pb.h"
#include "proto/predict_service.grpc.pb.h"


typedef google::protobuf::Map<std::string, serving::TensorProto> ArgrmentMap;


class Predictor {
public:
    virtual int init(const std::string& cfg, const std::string& weights) = 0;
    virtual int predict(const ArgrmentMap& input, ArgrmentMap* output) = 0;
    virtual int release() = 0;
};





class PredictionServiceImpl : public serving::PredictionService::Service  {


    std::map<std::string, std::shared_ptr<Predictor> > predictors_;

    // Service interface
public:
    void Init(const std::string& config_path);
    grpc::Status Predict(grpc::ServerContext *context, const serving::PredictRequest *request, serving::PredictResponse *response);
};
