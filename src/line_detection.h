#ifndef LINE_DETECTION_H_INCLUDED
#define LINE_DETECTION_H_INCLUDED

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Point3f> detectLines(cv::Mat image);

#endif // LINE_DETECTION_H_INCLUDED
