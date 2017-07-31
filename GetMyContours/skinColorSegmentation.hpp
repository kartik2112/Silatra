#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

Mat extractSkinColorRange(Mat& srcBGR,Mat& srcHSV,Mat& srcYCrCb);           //Entry point function
bool isInSkinRangeBGR(const u_char& B,const u_char& G,const u_char& R);
bool isInSkinRangeHSV(const u_char& H,const u_char& S,const u_char& V);
bool isInSkinRangeYCrCb(const u_char& Y, const u_char& Cr, const u_char& Cb);
void displayTrackbarsIfNeeded();

