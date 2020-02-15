#pragma once

#include "proto/predict.pb.h"
#include "proto/predict_service.grpc.pb.h"

class Predictor {

};





class PredictionServiceImpl : public serving::PredictionService::Service  {



    std::map<std::string, std::shared_ptr<Predictor> > predictors_;

    // Service interface
public:
    grpc::Status Predict(grpc::ServerContext *context, const serving::PredictRequest *request, serving::PredictResponse *response);
};
