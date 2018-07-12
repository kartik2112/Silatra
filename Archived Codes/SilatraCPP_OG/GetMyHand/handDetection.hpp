#ifndef __handDetection_hpp_INCLUDE__
#define __handDetection_hpp_INCLUDE__

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "skinColorSegmentation.hpp"
#include "trackBarHandling.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace cv;

Mat getMyHand(Mat& image);
Mat findHandContours(Mat& src);
Mat combineExtractedWithMain(Mat& maskedImg,Mat& image);
void prepareWindows();
void connectContours(vector<vector<Point> > &contours);
void reduceClusterPoints(vector< vector< Point > > &contours);

#endif
