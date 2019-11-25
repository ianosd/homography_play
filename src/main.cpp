#include <stdio.h>
#include <vector>
#include <tuple>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "line_detection.h"

using cv::Mat;
using cv::Point2f;
using cv::Point3d;
using cv::Point3f;

using std::cout;
using std::endl;
using std::pair;
using std::vector;

bool isInBounds(float x, float max){ return x >= 0 && x <=max; }


void drawLine(cv::Point3f homogenousLine, cv::Mat img)
{
    auto size = img.size();
    float leftYIntercept = -1;
    float rightYIntercept = -1;
    float topXIntercept = -1;
    float bottomXIntercept = -1;
    if (homogenousLine.y != 0)
    {
	leftYIntercept = -homogenousLine.z/homogenousLine.y;
	rightYIntercept = -(homogenousLine.x*size.width + homogenousLine.z)/homogenousLine.y;
    }

    if (homogenousLine.x != 0) 
    {
	topXIntercept = -homogenousLine.z/homogenousLine.x;
	bottomXIntercept = -(homogenousLine.y*size.height + homogenousLine.z)/homogenousLine.x;
    }

    vector<Point2f> ends;
    int i = 0;
    while (i < 4 && ends.size() < 2)
    {
	switch(i){
	    case 0:
		if (isInBounds(leftYIntercept, size.height))
		    ends.push_back(Point2f(0, leftYIntercept));
		break;
	    case 1:
		if (isInBounds(rightYIntercept, size.height))
		    ends.push_back(Point2f(size.width, rightYIntercept));
		break;
	    case 2:
		if (isInBounds(topXIntercept, size.width))
		    ends.push_back(Point2f(topXIntercept, 0));
		break;
	    case 3:
		if (isInBounds(bottomXIntercept, size.width))
		    ends.push_back(Point2f(bottomXIntercept, size.height));
		break;
	}
	i++;
    }
    if (ends.size() != 2)
    {
	std::cerr << "Request to draw line that is not in the image" << std::endl;
	return;
    }

    cv::line(img, ends[0], ends[1], cv::Scalar(0, 255, 0));
}

/**
 * Returns a matrix that maps the points pointsToMap to corresponding pointImages
 * The points use homogenous representation. That is, if it is to represent the point (x, y)
 * in an image, the point should be (alpha*x, alpha*y, alpha)
 */
Mat findHomographyFromPointCorespondences(const vector<Point3d>& pointsToMap, const vector<Point3d>& pointImages)
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

/**
 * Returns a matrix that maps the lines linesToMap to corresponding lineImages
 * Lines are represented as triplets (Point3d objects), Given a triplet l,
 * it has the usual meaning that l.t() * p = 0 for the (homogenously represented)
 * points on the line.
 */
Mat findHomographyFromLineCorrespondences(const vector<Point3d>& linesToMap, const vector<Point3d>& lineImages)
{
    // We can use the same procedure as when using points,
    // but we have to invert the inputs and transpose the output
    return findHomographyFromPointCorespondences(lineImages, linesToMap).t();
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

    Mat src;
    src = cv::imread(argv[1], 1);

    if (!src.data)
    {
        printf("No src data \n");
        return -1;
    }

    auto lines = detectLines(src);

    for (auto line : lines)
    {
	cout << line.x << ", " << line.y << ", " << line.z << std::endl;
    }

    for (int i = 0; i< 10; i++){
	drawLine(lines[i], src);
    }

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, src);
    cv::waitKey(0);

    return 0;
}
