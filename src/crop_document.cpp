#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "json.hpp"

using cv::Mat;
using cv::Point2f;
using cv::Point3d;
using cv::Point3f;
using cv::Vec2f;
using cv::Vec4f;
using cv::Scalar;
using cv::Size;

using std::cerr;
using std::cout;
using std::endl;
using std::pair;
using std::vector;

// get the scaling factor to be applied to an image so that it has at most a maximum width
double getOutputScaling(const Mat& image, int maxWidth){
    if (image.empty())
    {
        throw std::invalid_argument("Empty image provided!");
    }

    if (image.cols <= maxWidth)
    {
        return 1;
    }

    return static_cast<double>(maxWidth) / image.cols;
}

// convert a line segment (x1, y1, x2, y2) to the coefficients a, b, c defining a line by
// a*x + b*y + c = 0
Point3f segmentToHomogenousLine(const Vec4f& segment) {
    return {segment[1] - segment[3], segment[2] - segment[0], segment[0]*segment[3] - segment[1]*segment[2]};
}

// constants used by the program
struct Config
{
    int max_width; // the maximum display width of an output image
    int paper_width; // the width and height of the resulting image
    int paper_height; 
    double segment_range; // minimum range in pixels for line segment selection
};

const char *LINE_SELECTION_WINDOW_NAME = "Line selection";
const std::string edgeNames[] = {"top", "bottom", "left", "right"};
Config global_config;

// Define a function to populate Config from JSON
void from_json(const nlohmann::json &j, Config &config)
{
    j.at("max_width").get_to(config.max_width);
    j.at("segment_range").get_to(config.segment_range);
    j.at("paper_width").get_to(config.paper_width);
    j.at("paper_height").get_to(config.paper_height);
}

// Function to read a JSON file into a nlohmann::json object
bool readJsonFile(const std::string &filePath, nlohmann::json &jsonData)
{
    try
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            cerr << "Error: Could not open file " << filePath << std::endl;
            return false;
        }

        file >> jsonData;
        return true;
    }
    catch (const std::exception &e)
    {
        cerr << "Error reading file: " << e.what() << std::endl;
        return false;
    }
}

bool readConfig()
{
    nlohmann::json data;
    if (!readJsonFile("config.json", data))
    {
        return false;
    }
    try
    {
        global_config = data.get<Config>();
        return true;
    }
    catch (const nlohmann::json::out_of_range &e)
    {
        cerr << "Error in config file: " << e.what() << endl;
        return false;
    }
}

/**
 * Returns a matrix that maps the points pointsToMap to corresponding pointImages
 * The points use homogenous representation. That is, if it is to represent the point (x, y)
 * in an image, the point should be (alpha*x, alpha*y, alpha)
 */
Mat findHomographyFromPointCorespondences(const vector<Point3d> &pointsToMap, const vector<Point3d> &pointImages)
{
    Mat A(pointsToMap.size() * 2, 9, CV_64F);
    for (int i = 0; i < pointsToMap.size(); i++)
    {
        const Point3d &p0 = pointsToMap[i];
        const Point3d &p1 = pointImages[i];
        Mat r0 = (cv::Mat_<double>(1, 9) << p0.x * p1.z, p0.y * p1.z, p0.z * p1.z, 0, 0, 0, -p0.x * p1.x, -p0.y * p1.x, -p0.z * p1.x);
        r0.copyTo(A.row(2 * i));
        Mat r1 = (cv::Mat_<double>(1, 9) << 0, 0, 0, p0.x * p1.z, p0.y * p1.z, p0.z * p1.z, -p0.x * p1.y, -p0.y * p1.y, -p0.z * p1.y);
        r1.copyTo(A.row(2 * i + 1));
    }
    cout << A << endl;
    auto res = cv::SVD(A, cv::SVD::FULL_UV);
    cout << res.vt << endl
         << res.w << endl;
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
Mat findHomographyFromLineCorrespondences(const vector<Point3d> &linesToMap, const vector<Point3d> &lineImages)
{
    // We can use the same procedure as when using points,
    // but we have to invert the inputs and transpose the output
    return findHomographyFromPointCorespondences(lineImages, linesToMap).t();
}

struct DocumentEdges
{
    Point3d top;
    Point3d bottom;
    Point3d left;
    Point3d right;

    DocumentEdges() = default;

    // Constructor to initialize from a vector
    DocumentEdges(const vector<Point3f>& points) {
        if (points.size() != 4) {
            throw std::invalid_argument("Expected exactly 4 points to initialize DocumentEdges");
        }

        top = points[0];
        bottom = points[1];
        left = points[2];
        right = points[3];
    };

    // Constructor to initialize from an initializer list
    DocumentEdges(std::initializer_list<Point3d> points) {
        if (points.size() != 4) {
            throw std::invalid_argument("Expected exactly 4 points in initializer list to initialize DocumentEdges");
        }

        auto it = points.begin();
        top = *it++;
        bottom = *it++;
        left = *it++;
        right = *it;
    }

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
               << "Right: " << edges.right << endl
               << "---------------" << endl;
}

// Function to calculate the squared distance from a point to a segment
double pointToSegmentDistanceSquared(const Point2f &p, const Vec4f &lineSegment)
{
    Point2f p1(lineSegment[0], lineSegment[1]); // Start of the segment
    Point2f p2(lineSegment[2], lineSegment[3]); // End of the segment

    // Vector from p1 to p2
    Point2f v = p2 - p1;

    // Vector from p1 to the point
    Point2f w = p - p1;

    // Project w onto v, normalized by the length squared of v
    double t = (w.dot(v)) / (v.dot(v));

    // Clamp t to the range [0, 1] to stay within the segment
    t = std::max(0.0, std::min(1.0, t));

    // Compute the projection point on the segment
    Point2f projection = p1 + t * v;

    // Return squared distance from p to the projection point
    return (p - projection).dot(p - projection);
}

// Function to find the closest line segment to a given point
std::optional<Vec4f> findClosestSegment(const Point2f &point, const vector<Vec4f> &lineSegments, double maxDistance)
{
    double minDistance = maxDistance;
    std::optional<Vec4f> result;

    for (const auto &segment : lineSegments)
    {
        double dist = std::sqrt(pointToSegmentDistanceSquared(point, segment));
        if (dist < minDistance)
        {
            minDistance = dist;
            result.emplace(segment);
        }
    }

    return result;
}

// Contains the logic for selecting lines in an image
class LineSelector
{
public:
    LineSelector(const Mat &srcImage, double segmentRange): segmentRange(segmentRange)
    {
        Mat srcGray;
        cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
        detector->detect(srcGray, allSegments);
    }

    std::optional<Vec4f> selectLineNearPoint(Point2f point)
    {
        if (hasAllEdges()){
            throw std::logic_error("selectLineNearPoint should only be called if hasAllEdges is false!");
        }

        auto result = findClosestSegment(point, this->allSegments, segmentRange);
        if (result.has_value()) {
            selectedSegments.push_back(result.value());
        }
        return result;
    }

    void clearSelection() {
        selectedSegments.clear();
    }

    bool hasAllEdges() const {
        return selectedSegments.size() == 4;
    }

    int getCurrentEdgeIndex() const {
        return selectedSegments.size();

    }

    DocumentEdges getDocumentEdges() const {
        vector<Point3f> homogenousEdges(4);
        std::transform(selectedSegments.begin(), selectedSegments.end(), homogenousEdges.begin(),
                       &segmentToHomogenousLine);
        return DocumentEdges(homogenousEdges);
    }

    const vector<Vec4f>& getAllSegments() const{
        return this->allSegments;
    }

private:
    vector<Vec4f> selectedSegments;
    vector<Vec4f> allSegments;
    double segmentRange;
};

// gather together the objects to deal with when a mouse event happens
class MouseDataWrapper {
public:
        MouseDataWrapper(LineSelector& selector, double scale, Mat& outputImage): selector(selector), scale(scale), outputImage(outputImage) {}
        static void onMouse(int event, int x, int y, int, void* data){
            if( event != cv::EVENT_LBUTTONDOWN )
                return;
            
            MouseDataWrapper* connector = reinterpret_cast<MouseDataWrapper*>(data);
            Point2f point(x, y);

            if (connector->selector.hasAllEdges()){
                cout << "All edges have been selected! Accept using 'a' or start over using 'd'." << endl;
                return;
            }
            auto line = connector->selector.selectLineNearPoint(point/connector->scale);
            if (line.has_value()){
                auto segment = line.value()*connector->scale;
                cv::line(connector->outputImage, Point2f(segment[0], segment[1]), Point2f(segment[2], segment[3]), Scalar(0, 255, 0), 2);
                if (!connector->selector.hasAllEdges()){
                    cout << "Currently selecting: " << edgeNames[connector->selector.getCurrentEdgeIndex()] << endl;
                }
            } else {
                cout << "No line close enough!" << endl;
            }
        }
private:
    double scale;
    Mat& outputImage;
    LineSelector& selector;
};

// Overload the multiplication operator
Size operator*(const Size& size, double scale) {
    return Size(static_cast<int>(size.width * scale), 
                static_cast<int>(size.height * scale));
}

// Overload the multiplication operator for reversed order
Size operator*(double scale, const Size& size) {
    return size * scale; // Reuse the above implementation
}

Mat drawResizedImageAndSegments(const Mat& src, double scale, const vector<Vec4f> lineSegments){
    Mat resizedSrc;
    resize(src, resizedSrc, src.size()*scale, 0, 0, cv::INTER_LINEAR);

    // resize the detected segments for drawing on the user input image
    vector<Vec4f> resizedSegments(lineSegments.size());
    std::transform(lineSegments.begin(), lineSegments.end(),
                   resizedSegments.begin(),
                   [scale](const Vec4f &v) { return v * scale; });

    for (auto segment : resizedSegments) {
        cv::line(resizedSrc, Point2f(segment[0], segment[1]), Point2f(segment[2], segment[3]), Scalar(255, 0, 0));
    }

    return resizedSrc;
}

std::optional<DocumentEdges> getDocumentEdges(const Mat& src) {
    // If the input image is too large, resize it for user input operations
    double scale = getOutputScaling(src, global_config.max_width);

    LineSelector selector(src, global_config.segment_range/scale);
    if (selector.getAllSegments().empty()) {
        cerr << "No line segments detected!";
        return std::make_optional<DocumentEdges>();
    }

    Mat resizedSrc = drawResizedImageAndSegments(src, scale, selector.getAllSegments());

    MouseDataWrapper connector(selector, scale, resizedSrc);
    cv::namedWindow(LINE_SELECTION_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(LINE_SELECTION_WINDOW_NAME, MouseDataWrapper::onMouse, &connector);

    cout << "Select 4 line segments representing the document bounds (top, bottom, left, right), by clicking (near) them."
         << endl << "Use 'q' to cancel the process, and 'a' to accept the selection." << endl;

    // TODO duplicate code
    cout << "Currently selecting: " << edgeNames[selector.getCurrentEdgeIndex()] << endl;
    while (true)
    {
        cv::imshow(LINE_SELECTION_WINDOW_NAME, resizedSrc);

        switch (cv::waitKey(30))
        {
            case 'q':
                return std::make_optional<DocumentEdges>();
            case 'a':
                if (selector.hasAllEdges()){
                    return selector.getDocumentEdges();
                } else {
                    cout << "4 segments need to be selected!" << endl;
                    break;
                }
            case 'd':
                selector.clearSelection();
                // TODO duplicate code
                resizedSrc = drawResizedImageAndSegments(src, scale, selector.getAllSegments());
                break;
        }
    }

    cv::destroyWindow(LINE_SELECTION_WINDOW_NAME);
    return std::make_optional<DocumentEdges>();
}

Mat crop(const Mat& src, const DocumentEdges& edges) {
    DocumentEdges paperFrameEdges{Point3d(0, 1, 0), Point3d(0, -1, global_config.paper_height), Point3d(1, 0, 0), Point3d(-1, 0, global_config.paper_width)};
    auto homography = findHomographyFromLineCorrespondences(edges.toVector(), paperFrameEdges.toVector());
    Mat result(global_config.paper_width, global_config.paper_height, src.type());
    cv::warpPerspective(src, result, homography, {global_config.paper_width, global_config.paper_height});
    return result;
}

int main(int argc, char **argv) {
    if (!readConfig()) {
        cerr << "Could not read config.json" << endl;
        return -1;
    }

    if (argc < 2) {
        cerr << "usage: crop_document <Image_Path> [<Output Path>]" << endl;
        return -1;
    }

    Mat src;
    src = cv::imread(argv[1], 1);

    if (!src.data) {
        cerr << "No src data" << endl;
        return -1;
    }

    std::optional<DocumentEdges> edges = getDocumentEdges(src);
    if (!edges.has_value()) {
        cout << "No edges selected. Exiting." << endl;
        return -1;
    }

    Mat cropped = crop(src, edges.value());
    if (argc == 3) {
        cout << "Saving result to " << argv[2] << endl;
        cv::imwrite(argv[2], cropped);
    }
    else {
        cv::imshow("Cropped Image", cropped);
        cv::waitKey(0);
    }

    return 0;
}