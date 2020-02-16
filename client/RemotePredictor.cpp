#include "RemotePredictor.h"
#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include "proto/predict.pb.h"
#include "proto/predict_service.grpc.pb.h"

using namespace serving;

int RemoteInfo::init(const std::string &ipaddr, int port)
{
    this->ipaddr = ipaddr;
    this->port = port;
    return 0;
}

struct RemotePredictor::Impl {
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<PredictionService::Stub> stub;
    grpc::ClientContext context;
};

RemotePredictor::RemotePredictor() {
    impl_ = new Impl;
}

RemotePredictor::~RemotePredictor() {
    delete impl_;
}

int RemotePredictor::init(const RemoteInfo &ri, const std::string &id)
{
    std::string target = ri.ipaddr + ":" + std::to_string(ri.port);
    impl_->channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    impl_->stub = PredictionService::NewStub(impl_->channel);
}


typedef google::protobuf::Map<std::string, serving::TensorProto> ArgumentMap;

int RemotePredictor::predict(int w, int h, int c, void *data)
{
    PredictRequest request;

    ArgumentMap* am = request.mutable_inputs();

    am->clear();
    TensorProto& tp = (*am)["input"];
    tp.set_dtype(serving::DT_FLOAT);
    tp.mutable_tensor_shape()->add_dim()->set_size(1);
    tp.mutable_tensor_shape()->add_dim()->set_size(c);
    tp.mutable_tensor_shape()->add_dim()->set_size(h);
    tp.mutable_tensor_shape()->add_dim()->set_size(w);
    memcpy(tp.mutable_float_val()->mutable_data(), data, sizeof(float)*c*h*w);

    PredictResponse response;
    impl_->stub->Predict(&impl_->context, request, &response);

}

int RemotePredictor::release()
{

}

