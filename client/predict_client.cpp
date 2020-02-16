#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include "proto/predict.pb.h"
#include "proto/predict_service.grpc.pb.h"
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{

    cv::VideoCapture vc(argv[1]);


    using namespace serving;
    typedef google::protobuf::Map<std::string, serving::TensorProto> ArgumentMap;

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:8051", grpc::InsecureChannelCredentials());
    std::unique_ptr<PredictionService::Stub> stub = PredictionService::NewStub(channel);
    grpc::ClientContext context;


    int count = 0;
    while (1) {


        /// fetch image
        cv::Mat mat;
        vc >> mat;

        if (mat.empty()) {
            break;
        }


        /// preprosess input
        cv::Mat inImage;
        {
            cv::Mat rszImage;
            cv::resize(mat, rszImage, cv::Size(320, 240));

            //cv::cvtColor(rszImage, rszImage, CV_BGR2RGB);


            inImage.create(rszImage.size(), CV_32FC3);
            int C = inImage.channels();
            int H = inImage.rows;
            int W = inImage.cols;
            uchar* src_ptr = rszImage.data;
            float* dst_ptr = (float*)inImage.data;
            for (int c=0; c<C; c++) {
                for (int h=0; h<H; h++) {
                    for (int w=0; w<W; w++) {
                        // caffe
                        dst_ptr[c*H*W+h*W+w] = float(src_ptr[h*W*C+w*C+c]) / 255.f;
                        // tf
                        //dst_ptr[h*W*C+w*C+c] = float(src_ptr[h*W*C+w*C+c]) / 255.f;
                    }
                }
            }
        }

        /// fill request
        PredictRequest request;
        {
            request.mutable_model_spec()->set_name("MNN");
            ArgumentMap* am = request.mutable_inputs();

            am->clear();
            TensorProto& tp = (*am)["data"];
            tp.set_dtype(serving::DT_FLOAT);
            tp.mutable_tensor_shape()->add_dim()->set_size(1);
            tp.mutable_tensor_shape()->add_dim()->set_size(inImage.channels());
            tp.mutable_tensor_shape()->add_dim()->set_size(inImage.rows);
            tp.mutable_tensor_shape()->add_dim()->set_size(inImage.cols);
            int64_t length = inImage.channels()*inImage.rows*inImage.cols;
            tp.mutable_float_val()->Resize(length, 0);
            memcpy(tp.mutable_float_val()->mutable_data(), inImage.data, sizeof(float)*length);
        }

        /// remote call Predict
        PredictResponse response;
        {
            stub->Predict(&context, request, &response);
        }

        /// postprocess output
        {
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

        count++;
    }

    return 0;
}
