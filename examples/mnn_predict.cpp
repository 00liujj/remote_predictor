#include "MNN/Tensor.hpp"
#include "MNN/Interpreter.hpp"
#include <memory>
#include <opencv2/opencv.hpp>


int main(int argc, char *argv[])
{

    using namespace MNN;

    if (argc < 3) {
        printf("Usage: %s model.mnn {input.jpg|input.mp4}\n", argv[0]);
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    config.numThread = 4;


    Session* session = net->createSession(config);

    Tensor* inputTensor  = net->getSessionInput(session, NULL);
    Tensor* outputTensor = net->getSessionOutput(session, NULL);

    cv::VideoCapture vc(argv[2]);


    char* env = getenv("OUT_TYPE");
    std::string out_type;
    if (env) {
        out_type = env;
    }



    int count = 0;
    while (1) {

        cv::Mat mat;


        vc >> mat;

        if (mat.empty()) {
            break;
        }
        //image preproccess

        int netH = inputTensor->height();
        int netW = inputTensor->width();
        int netC = inputTensor->channel();


        cv::Mat rszImage;
        cv::resize(mat, rszImage, cv::Size(netW, netH));

        cv::Mat inImage;
        rszImage.convertTo(inImage, CV_32F, 1.f/255.f);

        //cv::cvtColor(inImage, inImage, CV_BGR2RGB);

        Tensor inputTensorUser(inputTensor, Tensor::DimensionType::TENSORFLOW);
        memcpy(inputTensorUser.host<float>(), inImage.data, sizeof(float)*netH*netW*netC);


        cv::TickMeter tm;
        //run
        inputTensor->copyFromHostTensor(&inputTensorUser);
        printf("the input shape is nchw %dx%dx%dx%d\n", inputTensor->batch(), inputTensor->channel(),
               inputTensor->height(), inputTensor->width());

        tm.start();
        net->runSession(session);
        tm.stop();


        Tensor outputTensorUser(outputTensor, outputTensor->getDimensionType());
        outputTensor->copyToHostTensor(&outputTensorUser);


        //Tensor& out = *outputTensor;
        Tensor& out = outputTensorUser;

        printf("the output shape is nchw %dx%dx%dx%d, time %f ms\n", out.batch(),
               out.channel(), out.height(), out.width(), tm.getTimeMilli());

        // output
        if (out_type == "DET") {
            int nboxes = out.height();
            printf("detection output: nboxes %d\n", nboxes);
            cv::Mat showImg = mat.clone();

            for (int b=0; b<nboxes; b++) {
                float* p = out.host<float>() + b * out.width();
                printf("box %d, label %d, score %f, bbox %f %f %f %f\n",
                       b, (int)p[0], p[1], p[2], p[3], p[4], p[5]);

                // draw
                cv::Rect rect(p[2]*showImg.cols, p[3]*showImg.rows, (p[4]-p[2])*showImg.cols, (p[5]-p[3])*showImg.rows);
                cv::rectangle(showImg, rect, CV_RGB(255,0,0));
                std::string txt = cv::format("%d-%.2f", (int)p[0], p[1]);
                cv::putText(showImg, txt, rect.tl(), 1, 1, CV_RGB(255,0,0));
            }

            std::string outfn = cv::format("det-image-%04d.jpg", count);
            cv::imwrite(outfn, showImg);
        }



        count++;
    }



    return 0;
}
