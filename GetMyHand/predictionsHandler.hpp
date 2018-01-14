#ifndef __predictionsHandler_hpp_INCLUDE__
#define __predictionsHandler_hpp_INCLUDE__

#include <iostream>
#include <deque>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void addPredictionToQueue(long long predictedSign);
long long predictSign();
void displaySignOnImage(long long predictSign);

#endif
