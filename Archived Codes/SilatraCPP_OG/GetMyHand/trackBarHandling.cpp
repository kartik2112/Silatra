/**
* To not display any trackbars, you can remove the function calls from handDetection.cpp, skinColorSegmentation.cpp
* or comment the contents of these functions
*/


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "handDetection.hpp"

using namespace cv;

void displayHandDetectionTrackbarsIfNeeded(Mat &image);
void displaySkinColorDetectionTrackbarsIfNeeded();
void onTrackbarChange(int,void *);



Mat *imagePtr;
extern int morphOpenKernSize,morphCloseKernSize,morphCloseNoOfIterations,thresh,kernSize;

extern int YMax,YMin,CrMax,CrMin,CbMax,CbMin;
extern int lH,hH,lS,hS,lV,hV;
extern int Rlow,Glow,Blow,gap,Rhigh,Ghigh,Bhigh;


/* To remove any trackbars, comment out those parts */
void displayHandDetectionTrackbarsIfNeeded(Mat &image){
	imagePtr = &image;
	
	namedWindow("Blurring, Contouring Controllers",WINDOW_AUTOSIZE);
	createTrackbar("Blur kernel size","Blurring, Contouring Controllers",&kernSize,10);
	
	createTrackbar("MORPH OPEN size","Blurring, Contouring Controllers",&morphOpenKernSize,25);
	createTrackbar("MORPH CLOSE size","Blurring, Contouring Controllers",&morphCloseKernSize,25);
	createTrackbar("MORPH OPEN No Of Iterations size","Blurring, Contouring Controllers",&morphCloseNoOfIterations,25);
	
	createTrackbar("Contours Threshold","Blurring, Contouring Controllers",&thresh,255);
	
	
	
	/*
	createTrackbar("Blur kernel size","Blurring, Contouring Controllers",&kernSize,10,onTrackbarChange);
	
	createTrackbar("MORPH OPEN size","Blurring, Contouring Controllers",&morphOpenKernSize,25,onTrackbarChange);
	createTrackbar("MORPH CLOSE size","Blurring, Contouring Controllers",&morphCloseKernSize,25,onTrackbarChange);
	createTrackbar("MORPH OPEN No Of Iterations size","Blurring, Contouring Controllers",&morphCloseNoOfIterations,25,onTrackbarChange);
	
	createTrackbar("Contours Threshold","Blurring, Contouring Controllers",&thresh,255,onTrackbarChange);
	*/
}


/**
* Comment the contents of this function if trackbars need not be displayed
*/
void displaySkinColorDetectionTrackbarsIfNeeded(){

	namedWindow("Skin Color Segmentation Controllers",WINDOW_AUTOSIZE);	
	
	createTrackbar("Low Hue","Skin Color Segmentation Controllers",&lH,180);
	createTrackbar("High Hue","Skin Color Segmentation Controllers",&hH,180);
	createTrackbar("Low Saturation","Skin Color Segmentation Controllers",&lS,255);
	createTrackbar("High Saturation","Skin Color Segmentation Controllers",&hS,255);
	createTrackbar("Low Value","Skin Color Segmentation Controllers",&lV,255);
	createTrackbar("High Value","Skin Color Segmentation Controllers",&hV,255);

	createTrackbar("YMax","Skin Color Segmentation Controllers",&YMax,255);
	createTrackbar("YMin","Skin Color Segmentation Controllers",&YMin,255);
	createTrackbar("CrMax","Skin Color Segmentation Controllers",&CrMax,255);
	createTrackbar("CrMin","Skin Color Segmentation Controllers",&CrMin,255);
	createTrackbar("CbMax","Skin Color Segmentation Controllers",&CbMax,255);
	createTrackbar("CbMin","Skin Color Segmentation Controllers",&CbMin,255);
	
	createTrackbar("I - R low","Skin Color Segmentation Controllers",&Rlow,255);
	createTrackbar("I - G low","Skin Color Segmentation Controllers",&Glow,255);
	createTrackbar("I - B low","Skin Color Segmentation Controllers",&Blow,255);
	createTrackbar("I - gap"  ,"Skin Color Segmentation Controllers",&gap,255);
	createTrackbar("II - R high","Skin Color Segmentation Controllers",&Rhigh,255);
	createTrackbar("II - G high","Skin Color Segmentation Controllers",&Ghigh,255);
	createTrackbar("II - B high","Skin Color Segmentation Controllers",&Bhigh,255);
	
	
	
	/*
	createTrackbar("Low Hue","Skin Color Segmentation Controllers",&lH,180,onTrackbarChange);
	createTrackbar("High Hue","Skin Color Segmentation Controllers",&hH,180,onTrackbarChange);
	createTrackbar("Low Saturation","Skin Color Segmentation Controllers",&lS,255,onTrackbarChange);
	createTrackbar("High Saturation","Skin Color Segmentation Controllers",&hS,255,onTrackbarChange);
	createTrackbar("Low Value","Skin Color Segmentation Controllers",&lV,255,onTrackbarChange);
	createTrackbar("High Value","Skin Color Segmentation Controllers",&hV,255,onTrackbarChange);
	
	createTrackbar("I - R low","Skin Color Segmentation Controllers",&Rlow,255,onTrackbarChange);
	createTrackbar("I - G low","Skin Color Segmentation Controllers",&Glow,255,onTrackbarChange);
	createTrackbar("I - B low","Skin Color Segmentation Controllers",&Blow,255,onTrackbarChange);
	createTrackbar("I - gap"  ,"Skin Color Segmentation Controllers",&gap,255,onTrackbarChange);
	createTrackbar("II - R high","Skin Color Segmentation Controllers",&Rhigh,255,onTrackbarChange);
	createTrackbar("II - G high","Skin Color Segmentation Controllers",&Ghigh,255,onTrackbarChange);
	createTrackbar("II - B high","Skin Color Segmentation Controllers",&Bhigh,255,onTrackbarChange);
	*/
}


/**
* Callback function to created trackbars
*/
void onTrackbarChange(int,void *){
	getMyHand(*imagePtr);
}
