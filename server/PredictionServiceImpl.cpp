#include "PredictionServiceImpl.h"
#include "MNNPredictor.h"

void PredictionServiceImpl::Init(const std::string& config_path)
{
    std::shared_ptr<MNNPredictor> pred(new MNNPredictor());
    pred->init("", "../ssd-model/demo.mnn");
    predictors_["MNN"] = pred;
}

grpc::Status PredictionServiceImpl::Predict(
        grpc::ServerContext *context,
        const serving::PredictRequest *request,
        serving::PredictResponse *response)
{
    std::string name = request->model_spec().name();
    auto iter = predictors_.find(name);

    std::cout << "client call Predict\n";

    if (iter != predictors_.end()) {
        std::shared_ptr<Predictor> pred = iter->second;
        int ret = pred->predict(request->inputs(), response->mutable_outputs());
        if (ret != 0) {
           return grpc::Status(grpc::UNAVAILABLE, "predictor " + name + " error");
        }
    } else {
        return grpc::Status(grpc::UNAVAILABLE, "can not find predictor with name " + name);
    }

    return grpc::Status::OK;
}
