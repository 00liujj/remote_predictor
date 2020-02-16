#include "RemotePredictor.h"
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    RemotePredictor pred;
    RemoteInfo ri;
    ri.init("localhost", 8051);
    pred.init(ri, "MNN");



    cv::VideoCapture vc(argv[1]);

    for (int i = 0; 1; ++i) {
        cv::Mat mat;
        vc >> mat;

        if (mat.empty()) {
            break;
        }


        cv::Mat src;
        cv::resize(mat, src, cv::Size(320, 240));


        cv::Mat_<float> dst(src.rows, src.cols);

        int C = src.channels();
        int H = src.rows;
        int W = src.cols;
        uchar* src_ptr = src.data;
        float* dst_ptr = (float*)dst.data;
        for (int c=0; c<C; c++) {
            for (int h=0; h<H; h++) {
                for (int w=0; w<W; w++) {
                    dst_ptr[c*H*W+h*W+w] = float(src_ptr[h*W*C+w*C+c]) / 255.f;
                }
            }
        }

        pred.predict(dst.cols, dst.rows, dst.channels(), dst.data);


    }



    pred.release();


    return 0;
}
