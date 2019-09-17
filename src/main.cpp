#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <tuple>

using cv::Mat;
using cv::Point2f;

using std::cout;
using std::endl;
using std::pair;
using std::vector;

Mat findHomographyFromLineCorrespondences(const vector<Point2f>& pointsToMapp, const vector<Point2f>& pointImages)
{
    return cv::findHomography(pointImages, pointsToMapp, 0).t();
}

const double A4Height = 141*4;
const double A4Width = 100*4;

struct DocumentEdges
{
    Point2f top;
    Point2f bottom;
    Point2f left;
    Point2f right;

    vector<Point2f> toVector() const
    {
        return {top, bottom, left, right};
    }
};

std::ostream &operator<<(std::ostream &out, const DocumentEdges &edges)
{
    return out << "Top: " << edges.top << endl
               << "Bottom: " << edges.bottom << endl
               << "Left: " << edges.left << endl
               << "Right: " << edges.right << "---------------" << endl;
}

const DocumentEdges A4Edges{Point2f(0, -0.1 / A4Height), Point2f(0, -0.9 / A4Height), Point2f(-0.1 / A4Width, 0), Point2f(-0.9 / A4Width, 0)};

DocumentEdges inputEdges;
pair<Point2f, Point2f> top_corners;
pair<Point2f, Point2f> bottom_corners;
pair<Point2f, Point2f> left_corners;
pair<Point2f, Point2f> right_corners;

pair<Point2f, Point2f> *const corner_pairs[] = {&top_corners, &left_corners, &bottom_corners, &right_corners};
pair<Point2f, Point2f> *const *current_corner_pair = &corner_pairs[0];

void Repaint()
{
}

Point2f ComputeLine(pair<Point2f, Point2f> corners)
{
    cv::Point3f a(corners.first.x, corners.first.y, 1.f);
    cv::Point3f b(corners.second.x, corners.second.y, 1.f);

    auto cross = a.cross(b);
    return Point2f(cross.x, cross.y) / cross.z;
}

void ComputeEdges()
{
    inputEdges = {ComputeLine(top_corners), ComputeLine(bottom_corners), ComputeLine(left_corners), ComputeLine(right_corners)};
}

void CallBackFunc(int event, int x, int y, int flags, void *userdata)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        (*current_corner_pair)->first = Point2f(x, y);
        break;
    case cv::EVENT_MOUSEMOVE:
        (*current_corner_pair)->second = Point2f(x, y);
        Repaint();
        break;
    case cv::EVENT_LBUTTONUP:
        (*current_corner_pair)->second = Point2f(x, y);
        ComputeEdges();
        Repaint();
        cout << "Selected element no " << std::distance(std::begin(corner_pairs), current_corner_pair) << endl;
        cout << "New input edges" << endl
             << inputEdges << endl;
        current_corner_pair++;
        if (current_corner_pair >= std::end(corner_pairs))
        {
            current_corner_pair = &corner_pairs[0];

            auto homography = findHomographyFromLineCorrespondences(inputEdges.toVector(), A4Edges.toVector());
            cout << "Computed following homography: " << homography << endl;
            Mat src = *(Mat*)userdata;
            Mat result(A4Width, A4Height, src.type());
            cv::warpPerspective(src, result, homography, {A4Width, A4Height});
            cv::imshow("result", result);
        }
        break;
    }
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    const char *window_name = "source image";

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

cv::namedWindow("result", CV_WINDOW_AUTOSIZE);
    cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback(window_name, CallBackFunc, &src);
    cv::imshow(window_name, src);
    cv::waitKey(0);

    return 0;
}
