#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"


#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

#include <experimental/filesystem>


using namespace std;
using namespace cv;

Mat getMyHand(Mat& image);
void detectAndEliminateFace(Mat frame);