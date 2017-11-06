#ifndef __trackBarHandling_hpp_INCLUDE__
#define __trackBarHandling_hpp_INCLUDE__

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "handDetection.hpp"

using namespace cv;

void displayHandDetectionTrackbarsIfNeeded(Mat &image);
void displaySkinColorDetectionTrackbarsIfNeeded();
void onTrackbarChange(int,void *);

#endif
