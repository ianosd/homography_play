#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat src;
    src = imread( argv[1], 1 );

    if ( !src.data )
    {
        printf("No src data \n");
        return -1;
    }

    Mat srcGray;
    ocl:cvtColor(src, srcGray, COLOR_BGR2GRAY);

    Mat sobelX;
    Sobel(srcGray, sobelX, srcGray.type(), 1, 0, 3, 1, 0, BORDER_REFLECT_101);

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", sobelX);

    waitKey(0);

    return 0;
}
