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




int morphOpenKernSize=2,morphCloseKernSize=3;
int morphCloseNoOfIterations=3;

int kernSize=2;
int thresh=100;
int contourDistThreshold = 30;
double startTime;

// extern int lH,lS,lV,hH,hS,hV;
extern string subDirName;

// extern vector<double> frameStepsTimes;

extern char** args_v;
extern int args_c;

extern Rect faceBBox;
extern bool faceFound;

extern bool wrapperModeOn;

extern long long predictedSign;

int YMax=255,YMin=0,CrMax=200,CrMin=137,CbMax=150,CbMin=100;


/*
This is the main entry point function of this file
*/
Mat getMyHand(Mat& imageOG){
	// startTime=(double)getTickCount();  //---Timing related part

	detectAndEliminateFace(imageOG);

	Mat image,imageYCrCb;
    
	GaussianBlur(imageOG,image,Size(2*kernSize+1,2*kernSize+1),0,0);
    
	medianBlur(image,image,2*kernSize+1);
	
	/* Convert BGR Image into YCrCb Image */
	cvtColor(image,imageYCrCb,CV_BGR2YCrCb);
    
    Mat dst;
	inRange(imageYCrCb,Scalar(YMin,CrMin,CbMin),Scalar(YMax,CrMax,CbMax),dst);


	// frameStepsTimes[ SKIN_COLOR_EXTRACTION ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	// startTime=(double)getTickCount();  //---Timing related part
	

	Mat morphOpenElement = getStructuringElement(MORPH_CROSS,Size(morphOpenKernSize*2+1,morphOpenKernSize*2+1),Point(morphOpenKernSize,morphOpenKernSize));
	Mat morphCloseElement = getStructuringElement(MORPH_CROSS,Size(morphCloseKernSize*2+1,morphCloseKernSize*2+1),Point(morphCloseKernSize,morphCloseKernSize));
	Mat dstEroded;
    
	morphologyEx(dst,dstEroded,MORPH_OPEN,morphOpenElement);
	
	/* 
	This will act make white areas larger then make them smaller
	i.e. this will do erode(dilate(img))
	Thus filling out missing spots in big white areas.
	 */
	morphologyEx(dstEroded,dstEroded,MORPH_CLOSE,morphCloseElement);
	
	
	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE,Size(3,3),Point(1,1));
	/* This will enlarge white areas */
	dilate(dstEroded,dstEroded,dilateElement,Point(-1,-1),1);
	

	// frameStepsTimes[ MORPHOLOGY_OPERATIONS ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	// startTime=(double)getTickCount();  //---Timing related part

	// Mat maskedImg;
	// cvtColor(dstEroded,dstEroded,CV_GRAY2BGR);
	// bitwise_and(dstEroded,imageOG,maskedImg);
	
	// Mat finImg=combineExtractedWithMain(maskedImg,image);
	// Mat finImg = dstEroded;

	// frameStepsTimes[ MODIFIED_IMAGE_GENERATION ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	// startTime=(double)getTickCount();  //---Timing related part
	
	// Mat contouredImg=findHandContours(finImg);

	// frameStepsTimes[ HAND_CONTOURS_GENERATION ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	// startTime=(double)getTickCount();  //---Timing related part
	
	/// Show in a window  
	// imshow("Contours", contouredImg );	
	// imwrite("./ContourImages/img.png",contouredImg);
	// imshow("Morphed Mask",dstEroded);
	// imshow("Masked Image",maskedImg);
	// imshow("Final Image",finImg);
	// imshow("HSV + BGR Mask",dst);
	// imshow("HSV Mask",dstHSV);



	// Py_SetProgramName();
	// Py_Initialize();
	// PyRun_SimpleString("print '\nThis is first successful Python Statement being run from C++' ");
	// Py_Finalize();
	



	return dstEroded;	
    // return maskedImg;
}


void detectAndEliminateFace(Mat frame){
	String face_cascade_name = "./HaarCascades/haarcascade_frontalface_default.xml";
	CascadeClassifier face_cascade;
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); };
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	long long maxArea = 0;
	int maxAreaRectInd = -1;

	for( size_t i = 0; i < faces.size(); i++ )
	{
		long long area1 = (faces[i].width * faces[i].height);
		if( area1 > maxArea){
			maxArea = area1;
			maxAreaRectInd = i;
		}
		// Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		// ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
		// rectangle(frame,faces[i],Scalar(0,0,0),-1);

		// Mat faceROI = frame_gray( faces[i] );
	}

	if(maxAreaRectInd==-1){
		return;
	}

	faceFound = true;
	faceBBox = faces[maxAreaRectInd];

	//Modify face rectangle to eliminate neck and cover bigger part of face so as to eliminate possibility of leaving out some skin area
	faceBBox.x -= 10;
	faceBBox.y -= 10;
	faceBBox.width += 20;
	faceBBox.height += 45;
	rectangle(frame,faceBBox,Scalar(0,0,0),-1);

	//-- Show what you got
	// imshow( "Framed", frame );
}