#pragma once
#include "Predictor.h"

class PredictionServiceImpl : public serving::PredictionService::Service  {


    std::map<std::string, std::shared_ptr<Predictor> > predictors_;

    // Service interface
public:
    void Init(const std::string& config_path);
    grpc::Status Predict(grpc::ServerContext *context, const serving::PredictRequest *request, serving::PredictResponse *response);
};
