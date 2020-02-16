#pragma once

#include "Predictor.h"

#include "MNN/Tensor.hpp"
#include "MNN/Interpreter.hpp"



class MNNPredictor : public Predictor {


    std::shared_ptr<MNN::Interpreter> net_;
    MNN::Session* session_;



    // Predictor interface
public:
    static int copyTensorProto2MNNTensor(const serving::TensorProto& rpc, MNN::Tensor* mnn);
    static int copyMNNTensor2TensorProto(const MNN::Tensor& mnn, serving::TensorProto* rpc);

    int init(const std::string &cfg, const std::string &weights);
    int predict(const ArgumentMap &input, ArgumentMap *output);
    int release();
};
