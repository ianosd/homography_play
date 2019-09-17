#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <tuple>

using cv::Mat;
using cv::Point2f;
using cv::Point3d;

using std::cout;
using std::endl;
using std::pair;
using std::vector;

Mat findHomographyHomogenousInput(const vector<Point3d>& pointsToMap, const vector<Point3d>& pointImages)
{
    Mat A(pointsToMap.size() * 2, 9, CV_64F);
    for (int i = 0; i < pointsToMap.size(); i++){
	const Point3d& p0 = pointsToMap[i];
	const Point3d& p1 = pointImages[i];
	Mat r0 = (cv::Mat_<double>(1, 9) << p0.x*p1.z , p0.y*p1.z , p0.z*p1.z , 0 , 0 , 0 , -p0.x*p1.x , -p0.y*p1.x , -p0.z*p1.x);
	r0.copyTo(A.row(2*i));
	Mat r1 = (cv::Mat_<double>(1, 9) << 0 , 0 , 0 , p0.x*p1.z , p0.y*p1.z , p0.z*p1.z , -p0.x*p1.y , -p0.y*p1.y , -p0.z*p1.y);
	r1.copyTo(A.row(2*i+1));
    }
    cout << A << endl;
    auto res = cv::SVD(A, cv::SVD::FULL_UV);
    cout << res.vt << endl << res.w << endl;
    auto r = res.vt.row(8);
    cout << "A*r = " << A * r.t() << endl;
    Mat real_res(3, 3, CV_64F);
    std::copy(r.begin<double>(), r.end<double>(), real_res.begin<double>());
    return real_res;
}

Mat findHomographyFromLineCorrespondences(const vector<Point3d>& linesToMap, const vector<Point3d>& lineImages)
{
    auto transposed = findHomographyHomogenousInput(lineImages, linesToMap);
    Mat matImages(3, 4, CV_64F);
    for (int i = 0; i<3; i++)
	for (int j = 0; j < 4; j++)
	    matImages.at<double>(i, j) = i == 0 ? lineImages[j].x : (i == 1 ? lineImages[j].y : lineImages[j].z);

    Mat matToMap(3, 4, CV_64F);
    for (int i = 0; i<3; i++)
	for (int j = 0; j < 4; j++)
	    matToMap.at<double>(i, j) = i == 0 ? linesToMap[j].x : (i == 1 ? linesToMap[j].y : linesToMap[j].z);


    cout <<transposed << endl << transposed.type() << " " << matImages.type() << endl;
    Mat result = transposed * matImages;

    cout << "To map " << matImages << endl << "Expected " << matToMap << endl << "Actual " << result << endl;
    return transposed.t();
}

const double A4Height = 141*4;
const double A4Width = 100*4;

struct DocumentEdges
{
    Point3d top;
    Point3d bottom;
    Point3d left;
    Point3d right;

    vector<Point3d> toVector() const
    {
        return {top, bottom, left, right};
    }
};

std::ostream &operator<<(std::ostream &out, const DocumentEdges &edges)
{
    return out << "Top: " << edges.top << endl
               << "Bottom: " << edges.bottom << endl
               << "Left: " << edges.left << endl
               << "Right: " << edges.right << endl << "---------------" << endl;
}

const DocumentEdges A4Edges{Point3d(0, 1, 0), Point3d(0, -1, A4Height), Point3d(1, 0, 0), Point3d(-1, 0, A4Width)};

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

Point3d ComputeLine(pair<Point2f, Point2f> corners)
{
    cv::Point3f a(corners.first.x, corners.first.y, 1.f);
    cv::Point3f b(corners.second.x, corners.second.y, 1.f);

    return a.cross(b);
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
	cout << Point2f(x, y) << endl;
        break;
    case cv::EVENT_MOUSEMOVE:
        (*current_corner_pair)->second = Point2f(x, y);
        Repaint();
        break;
    case cv::EVENT_LBUTTONUP:
	cout << Point2f(x, y) << endl;
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

	    auto ptest = Mat(cv::Point3d(top_corners.first.x, top_corners.first.y, 1.f));
	    Mat transform = homography * ptest;
	     transform = transform / transform.at<double>(2, 0);
	    cout << "Test " << ptest << " -> " << transform << endl;
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

    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    Mat edges;
    cv::Canny(src_gray, edges, low_threshold, high_threshold);

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window_name, CallBackFunc, &src);
    cv::imshow(window_name, src);
    cv::waitKey(0);

    return 0;
}
