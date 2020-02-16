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

    request.mutable_model_spec()->set_name("MNN");
    ArgumentMap* am = request.mutable_inputs();

    am->clear();
    TensorProto& tp = (*am)["data"];
    tp.set_dtype(serving::DT_FLOAT);
    tp.mutable_tensor_shape()->add_dim()->set_size(1);
    tp.mutable_tensor_shape()->add_dim()->set_size(c);
    tp.mutable_tensor_shape()->add_dim()->set_size(h);
    tp.mutable_tensor_shape()->add_dim()->set_size(w);
    int64_t length = c*h*w;
    tp.mutable_float_val()->Resize(length, 0);
    memcpy(tp.mutable_float_val()->mutable_data(), data, sizeof(float)*length);

    PredictResponse response;
    impl_->stub->Predict(&impl_->context, request, &response);



    auto iter = response.outputs().find("detection_out");
    if (iter != response.outputs().end()) {
        const TensorProto& tpout = iter->second;
        int nboxes = tpout.tensor_shape().dim(2).size();
        printf("detection output: nboxes %d\n", nboxes);

        for (int b=0; b<nboxes; b++) {
            const float* p = tpout.float_val().data() + b * 6;
            printf("box %d, label %d, score %f, bbox %f %f %f %f\n",
                   b, (int)p[0], p[1], p[2], p[3], p[4], p[5]);
        }
    } else {
        printf("can not find detection_out\n");
    }


}

int RemotePredictor::release()
{

}

