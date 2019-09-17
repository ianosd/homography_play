#include <stdio.h>
#include <opencv2/opencv.hpp>

using cv::Mat;
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    const char *window_name = "gradient";

    double low_threshold = 20;
    double high_threshold = 50;

    Mat src, src_gray;
    src = cv::imread(argv[1], 1);
        
    if (!src.data)
    {
        printf("No src data \n");
        return -1;
    }

    cv::cvtColor(src, src_gray, CV_BGR2GRAY);

    Mat edges;
    cv::Canny(src_gray, edges, low_threshold, high_threshold);

    cv::namedWindow("Edges", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Grayscale", CV_WINDOW_AUTOSIZE);

    cv::imshow("Grayscale", src_gray);

    cv::imshow("Edges", edges);
    cv::waitKey(0);
    return 0;
}
