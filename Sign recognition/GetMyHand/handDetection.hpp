#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

// #include "skinColorSegmentation.hpp"
// #include "trackBarHandling.hpp"
// #include "predictionsHandler.hpp"
// #include "Classification/classifyPythonAPI.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

#include <experimental/filesystem>


// #define OVERALL 0
// #define SKIN_COLOR_EXTRACTION 1
// #define MORPHOLOGY_OPERATIONS 2
// #define MODIFIED_IMAGE_GENERATION 3
// #define HAND_CONTOURS_GENERATION 4
// #define CONTOURS_PRE_PROCESSING 5
// #define CONTOURS_IMPROVEMENT 6
// #define CONTOURS_POST_PROCESSING 7
// #define CONTOUR_CLASSIFICATION_IN_PY 8

using namespace std;
using namespace cv;
// namespace fs = std::experimental::filesystem;

Mat getMyHand(Mat& image);
// Mat findHandContours(Mat& src);
// Mat combineExtractedWithMain(Mat& maskedImg,Mat& image);
// void prepareWindows();
// void connectContours(vector<vector<Point> > &contours);
// void reduceClusterPoints(vector< vector< Point > > &contours, vector<vector<Point> > &hull);
// void findClassUsingPythonModels( vector<float> &distVector );
void detectAndEliminateFace(Mat frame);