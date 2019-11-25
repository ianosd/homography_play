#include "line_detection.h"

#include <vector>
#include <opencv2/opencv.hpp>

#include <iostream>

using std::vector;
using cv::Point3f;
using cv::Mat;

namespace 
{
    const double LOW_THRESHOLD = 20;
    const double HIGH_THRESHOLD = 50;
    const double HOUGH_RHO = 1;
    const double HOUGH_THETA = M_PI/180;
    const int HOUGH_THRESHOLD = 10;

    Point3f houghLineToHomogenousLine(Point3f hough_line)
    {
	return Point3f(-sin(hough_line.y), cos(hough_line.y), hough_line.x);
    }

}

vector<Point3f> detectLines(Mat src)
{
    Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    Mat edges;
    cv::Canny(src_gray, edges, LOW_THRESHOLD, HIGH_THRESHOLD);

    vector<cv::Point3f> hough_result;
    cv::HoughLines(edges, hough_result, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD);

    vector<cv::Point3f> result(hough_result.size());
    std::transform(hough_result.begin(), hough_result.end(), result.begin(), &houghLineToHomogenousLine);
    return result;
}
