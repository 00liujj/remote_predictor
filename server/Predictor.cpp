#include "Predictor.h"
#include "grpcpp/server_builder.h"

#include "MNN/Tensor.hpp"
#include "MNN/Interpreter.hpp"

using namespace MNN;
class MNNPredictor : public Predictor {


    std::shared_ptr<Interpreter> net_;
    Session* session_;



    // Predictor interface
public:
    int init(const std::string &cfg, const std::string &weights) {

        net_.reset(Interpreter::createFromFile(weights.c_str()));

        ScheduleConfig config;
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        session_ = net_->createSession(config);
    }

    static int copyTensorProto2MNNTensor(const serving::TensorProto& rpc, Tensor* mnn) {
        if (rpc.dtype() != serving::DT_FLOAT) {
            std::cout << "rpc tensor not float\n";
            return -1;
        }
        if (rpc.tensor_shape().dim_size() != mnn->shape().size()) {
            std::cout << "tensor dim not equal\n";
            return -1;
        }

        int64_t length = 1;
        for (int i=0; i<rpc.tensor_shape().dim_size(); i++) {
            if (rpc.tensor_shape().dim(i).size() != mnn->shape()[i]) {
                std::cout << "tensor shape not match\n";
                return -1;
            }
            length *= rpc.tensor_shape().dim(i).size();
        }

        memcpy(mnn->host<float>(), rpc.float_val().data(), length*sizeof(float));
        return 0;
    }

    static int copyMNNTensor2TensorProto(const Tensor& mnn, serving::TensorProto* rpc, bool create_on_need) {

        rpc->clear_tensor_shape();
        rpc->clear_float_val();
        rpc->set_dtype(serving::DT_FLOAT);
        int64_t length = 1;
        for (int i=0; i<mnn.shape().size(); i++) {
            int d = mnn.shape()[i];
            length *= d;
            rpc->mutable_tensor_shape()->add_dim()->set_size(d);
        }
        rpc->mutable_float_val()->Resize(length, 0);
        memcpy(rpc->mutable_float_val()->mutable_data(), mnn.host<float>(), length*sizeof(float));
        return 0;
    }


    int predict(const ArgumentMap &input, ArgumentMap *output) {


        /// copy input to mnn
        auto mnn_input = net_->getSessionInputAll(session_);

        for (auto& it : mnn_input) {
            std::string name = it.first;
            auto iter = input.find(name);
            if (iter == input.end()) {
                std::cout << "can not find mnn input " << name << " in remote input" << std::endl;
                return -1;
            }
            const serving::TensorProto& rpc_tensor = iter->second;
            Tensor* mnn_tensor = it.second;
            copyTensorProto2MNNTensor(rpc_tensor, mnn_tensor);
        }

        /// run
        net_->runSession(session_);

        /// copy mnn to output
        auto mnn_output = net_->getSessionOutputAll(session_);

        output->clear();
        for (auto& it : mnn_output) {
            std::string name = it.first;
            Tensor* mnn_tensor = it.second;
            serving::TensorProto& rpc_tensor = (*output)[name];
            copyMNNTensor2TensorProto(*mnn_tensor, &rpc_tensor, true);
        }
    }
    int release() {
        net_->releaseSession(session_);
        net_.reset();
    }
};




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
    int ret = -1;
    std::string name = request->model_spec().name();
    auto iter = predictors_.find(name);
    if (iter != predictors_.end()) {
        std::shared_ptr<Predictor> pred = iter->second;
        ret = pred->predict(request->inputs(), response->mutable_outputs());
    }

    return grpc::Status::OK;
}


int main(int argc, char *argv[])
{

    std::string server_address("0.0.0.0:8051");
    PredictionServiceImpl service;

    service.Init("");

    grpc::ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();

    return 0;
}

