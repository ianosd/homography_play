#include <stdio.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "json.hpp"

using cv::Mat;
using cv::Point2f;
using cv::Point3d;
using cv::Point3f;

using std::cout;
using std::endl;
using std::pair;
using std::vector;

bool isInBounds(float x, float max){ return x >= 0 && x <=max; }

Mat resizeToMaxWidth(const Mat& image, int maxWidth, double& scale) {
    if (image.empty()) {
        std::cerr << "Error: Empty image provided" << endl;
        return Mat();
    }

    // Check if resizing is needed
    if (image.cols <= maxWidth) {
        return image.clone(); // No resizing needed, return a copy of the image
    }

    // Calculate the scaling factor to resize while maintaining the aspect ratio
    scale = static_cast<double>(maxWidth) / image.cols;

    // Compute the new dimensions
    int newWidth = maxWidth;
    int newHeight = static_cast<int>(image.rows * scale);

    // Resize the image
    Mat resizedImage;
    resize(image, resizedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    return resizedImage;
}

namespace 
{
    Point3f houghLineToHomogenousLine3(Point3f hough_line)
    {
        return Point3f(cos(hough_line.y), sin(hough_line.y), -hough_line.x);
    }

    Point3f houghLineToHomogenousLine(const cv::Vec2f& hough_line)
    {
        return Point3f(cos(hough_line[1]), sin(hough_line[1]), -hough_line[0]);
    }
}

struct Config {
    int hough_threshold;
    double low_threshold;
    double high_threshold;
    double hough_rho;
    double hough_theta;
};

Config global_config;

// Define a function to populate Config from JSON
void from_json(const nlohmann::json& j, Config& config) {
    j.at("hough_threshold").get_to(config.hough_threshold);
    j.at("low_threshold").get_to(config.low_threshold);
    j.at("high_threshold").get_to(config.high_threshold);
    j.at("hough_rho").get_to(config.hough_rho);
    j.at("hough_theta").get_to(config.hough_theta);
}

// Function to read a JSON file into a nlohmann::json object
bool readJsonFile(const std::string& filePath, nlohmann::json& jsonData) {
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            return false;
        }

        file >> jsonData;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading file: " << e.what() << std::endl;
        return false;
    }
}

bool readConfig(){
    nlohmann::json data;
    if (!readJsonFile("config.json", data)){
        return false;
    }
    global_config = data.get<Config>();
    return true;
}

void drawLine(cv::Point3f homogenousLine, cv::Mat img) {
    // Ensure the image is valid
    if (img.empty()) {
        std::cerr << "Error: Image is empty." << std::endl;
        return;
    }

    // Extract the line coefficients from the homogeneous representation
    float a = homogenousLine.x; // Coefficient for x
    float b = homogenousLine.y; // Coefficient for y
    float c = homogenousLine.z; // Constant term

    // Define two points that will represent the line within the image bounds
    cv::Point pt1, pt2;

    // Handle vertical lines (b ≈ 0)
    if (std::abs(b) > 1e-6) {
        // y = (-a*x - c) / b
        pt1 = cv::Point(0, static_cast<int>(-c / b)); // Point at x=0
        pt2 = cv::Point(img.cols - 1, static_cast<int>(-(a * (img.cols - 1) + c) / b)); // Point at x=img.cols-1
    } else if (std::abs(a) > 1e-6) {
        // Handle horizontal lines (a ≈ 0)
        // x = (-b*y - c) / a
        pt1 = cv::Point(static_cast<int>(-c / a), 0); // Point at y=0
        pt2 = cv::Point(static_cast<int>(-(b * (img.rows - 1) + c) / a), img.rows - 1); // Point at y=img.rows-1
    } else {
        // Degenerate line (a and b both near zero)
        std::cerr << "Error: Invalid homogeneous line coefficients." << std::endl;
        return;
    }

    // Clip the line to the image boundary using cv::clipLine
    if (cv::clipLine(img.size(), pt1, pt2)) {
        // Draw the line on the image
        cv::line(img, pt1, pt2, 255, 2); // Green line with thickness 2
    } else {
        std::cerr << "Warning: Line does not intersect the image." << std::endl;
    }
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

std::pair<vector<Point3f>, cv::Mat> detectLines(const cv::Mat& src)
{
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(src_gray, edges, global_config.low_threshold, global_config.high_threshold);

    vector<cv::Point3f> hough_result;
    cv::HoughLines(edges, hough_result, global_config.hough_rho, global_config.hough_theta, global_config.hough_threshold);

    vector<cv::Point3f> result(hough_result.size());
    std::transform(hough_result.begin(), hough_result.end(), result.begin(), &houghLineToHomogenousLine3);
    struct
    {
        bool operator()(Point3f a, Point3f b) const { return a.z > b.z; }
    }
    customLess;
    std::sort(result.begin(), result.end(), customLess);
    return std::make_pair(result, edges);
}

// Function to find the most represented line passing through a given point
cv::Vec2f findLine(const std::vector<cv::Vec2f>& lines, cv::Point2f point) {
    // Find the line most represented near the given point
    cv::Vec2f bestLine(0, 0);
    double minDistance = DBL_MAX;

    for (const auto& line : lines) {
        float rho = line[0];
        float theta = line[1];

        // Calculate the distance from the point to the line
        double a = cos(theta), b = sin(theta);
        double distance = abs(a * point.x + b * point.y - rho);

        if (distance < minDistance) {
            minDistance = distance;
            bestLine = line;
        }
    }

    return bestLine;
}

class LineSelector {
public:
    LineSelector(const std::vector<cv::Vec2f>& lines, const Mat& srcImage, double scaleArg) : lines(lines), scale(scaleArg), srcImage(srcImage)
    {
        outputImage = srcImage.clone();
    }

    // Custom logic
    void onMouse(int event, int x, int y) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            cout << "Mouse clicked at (" << x << ", " << y << ")" << endl;
            auto line = findLine(this->lines, cv::Point(x, y)/this->scale);
            selectedLines.push_back(line);
            cv::Vec2f scaledLine(line);
            scaledLine[0] *= this->scale;
            drawLine(houghLineToHomogenousLine(scaledLine), this->outputImage);
        }
    }

    // Static callback wrapper
    static void callback(int event, int x, int y, int flags, void* userdata) {
        auto* handler = reinterpret_cast<LineSelector*>(userdata);
        handler->onMouse(event, x, y);
    }

    const Mat& getOutputImage() const {
        return this->outputImage;
    }

private:
    const std::vector<cv::Vec2f>& lines;
    std::vector<cv::Vec2f> selectedLines;
    const Mat& srcImage;
    Mat outputImage;
    double scale;
};

void show(const char* name, const Mat& img){
    cv::imshow(name, img);
    cv::waitKey(0);
}

int main(int argc, char **argv)
{
    if (!readConfig()){
        printf("Could not read config.json");
        return -1;
    }
    if (argc != 2)
    {
        printf("usage: crop_document <Image_Path>\n");
        return -1;
    }
    Mat src0;
    src0 = cv::imread(argv[1], 1);
 
    if (!src0.data)
    {
        printf("No src data \n");
        return -1;
    }
    double scale0 = 1;
    Mat src = resizeToMaxWidth(src0, 800, scale0);
 
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(src_gray, edges, global_config.low_threshold, global_config.high_threshold);
    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();

    std::vector<cv::Vec4f> lines;
    detector->detect(src_gray, lines);

    detector->drawSegments(src, lines);
    show("Edges", src);

    vector<cv::Vec2f> houghLines;
    cv::HoughLines(edges, houghLines, global_config.hough_rho, global_config.hough_theta, global_config.hough_threshold);

    if (houghLines.empty()){
        std::cerr << "No lines detected!" << std::endl;
        return -1;
    }

    double scale = 1;
    cv::Mat scaledSrc = resizeToMaxWidth(src, 800, scale);
    cout << "Scale " << scale << std::endl;
 
    const char *output_image_name = "Line selection";
    cv::namedWindow(output_image_name, cv::WINDOW_AUTOSIZE);

    LineSelector lineSelector(houghLines, scaledSrc, scale);
    cv::setMouseCallback(output_image_name, LineSelector::callback, &lineSelector);

    while(true){
        cv::imshow(output_image_name, lineSelector.getOutputImage());

        if (cv::waitKey(30) == 27){
            break;
        }
    }
 
    return 0;
}