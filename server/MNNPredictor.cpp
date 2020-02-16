#include "MNNPredictor.h"

using namespace MNN;

int MNNPredictor::copyTensorProto2MNNTensor(const serving::TensorProto &rpc, Tensor *mnn) {
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

    Tensor buff(mnn, Tensor::DimensionType::CAFFE);

    memcpy(buff.host<float>(), rpc.float_val().data(), length*sizeof(float));

    mnn->copyFromHostTensor(&buff);

    return 0;
}

int MNNPredictor::copyMNNTensor2TensorProto(const Tensor &mnn, serving::TensorProto *rpc) {

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

    Tensor buff(&mnn, Tensor::DimensionType::CAFFE);
    mnn.copyToHostTensor(&buff);

    memcpy(rpc->mutable_float_val()->mutable_data(), buff.host<float>(), length*sizeof(float));
    return 0;
}

int MNNPredictor::init(const std::string &cfg, const std::string &weights) {

    net_.reset(Interpreter::createFromFile(weights.c_str()));

    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    config.numThread = 4;

    session_ = net_->createSession(config);
}

int MNNPredictor::predict(const ArgumentMap &input, ArgumentMap *output) {

    printf("call the mnn predict\n");
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
        copyMNNTensor2TensorProto(*mnn_tensor, &rpc_tensor);
    }
}

int MNNPredictor::release() {
    net_->releaseSession(session_);
    net_.reset();
}
