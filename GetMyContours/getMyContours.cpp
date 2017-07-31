/**
* This file contains the code that will dilate, erode the regions extracted based on color
*/


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "skinColorSegmentation.hpp"

#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

void getMyContours(Mat& image);
Mat findHandContours(Mat& src);
Mat combineExtractedWithMain(Mat& maskedImg,Mat& image);
void prepareTrackbarsNWindows();




int morphOpenKernSize=2,morphCloseKernSize=2;
int morphCloseNoOfIterations=3;

int kernSize=2;
int thresh=100;

extern int lH,lS,lV,hH,hS,hV;




void getMyContours(Mat& image){
	
	imshow("Original Image",image);
		
	//blur(image,image,Size(kernSize,kernSize),Point(-1,-1));
	GaussianBlur(image,image,Size(2*kernSize+1,2*kernSize+1),0,0);
	//medianBlur(image,image,2*kernSize+1);
	
	/* Convert BGR Image into HSV Image */
	Mat imageHSV,imageYCrCb;
	cvtColor(image,imageHSV,CV_BGR2HSV);	
	cvtColor(image,imageYCrCb,CV_BGR2YCrCb);
	//cvtColor(imageYCrCb,imageYCrCb,CV_YCrCb2BGR);
	imshow("YCrCb Im",imageYCrCb);

	Mat dstHSV;
	inRange(imageHSV,Scalar(lH,lS,lV),Scalar(hH,hS,hV),dstHSV);
	
	Mat dst=extractSkinColorRange(image,imageHSV,imageYCrCb);
	
	
	Mat morphOpenElement = getStructuringElement(MORPH_CROSS,Size(morphOpenKernSize*2+1,morphOpenKernSize*2+1),Point(morphOpenKernSize,morphOpenKernSize));
	Mat morphCloseElement = getStructuringElement(MORPH_CROSS,Size(morphCloseKernSize*2+1,morphCloseKernSize*2+1),Point(morphCloseKernSize,morphCloseKernSize));
	Mat dstEroded;
	
	//erode(dst,dstEroded,element);
	//dilate(dstEroded,dstEroded,element);
	
	//dilate(dstEroded,dstEroded,element);
	//erode(dstEroded,dstEroded,element);
	
	/* 
	This will act make white areas smaller then make them larger
	i.e. this will do dilate(erode(img))
	Thus removing noise.
	 */
	morphologyEx(dst,dstEroded,MORPH_OPEN,morphOpenElement);
	
	/* 
	This will act make white areas larger then make them smaller
	i.e. this will do erode(dilate(img))
	Thus filling out missing spots in big white areas.
	 */
	morphologyEx(dstEroded,dstEroded,MORPH_CLOSE,morphCloseElement);
	
	/* This will enlarge white areas */
	dilate(dstEroded,dstEroded,morphCloseElement,Point(-1,-1),morphCloseNoOfIterations);
	
	
	//cout<<dst.type()<<" "<<image.type()<<endl;
	
	
	Mat morphCloseElement1 = getStructuringElement(MORPH_ELLIPSE,Size(15,15),Point(7,7));
	/* AND this eroded mask with HSV */
	bitwise_and(dstEroded,dstHSV,dstEroded);
	
	morphologyEx(dstEroded,dstEroded,MORPH_CLOSE,morphCloseElement1);
	
	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE,Size(8,8),Point(4,4));
	/* This will enlarge white areas */
	dilate(dstEroded,dstEroded,dilateElement,Point(-1,-1),morphCloseNoOfIterations);
	//dilate(dstEroded,dstEroded,dilateElement,Point(-1,-1),morphCloseNoOfIterations);

	Mat maskedImg;
	cvtColor(dstEroded,dstEroded,CV_GRAY2BGR);
	bitwise_and(dstEroded,image,maskedImg);
	
	Mat finImg=combineExtractedWithMain(maskedImg,image);
	
	Mat contouredImg=findHandContours(finImg);
	
	/// Show in a window  
	imshow( "Contours", contouredImg );	
	imshow("Morphed Mask",dstEroded);
	imshow("Masked Image",maskedImg);
	imshow("Final Image",finImg);
	imshow("HSV + BGR Mask",dst);
	imshow("HSV Mask",dstHSV);
	
	
}


Mat combineExtractedWithMain(Mat& maskedImg,Mat& image){
	int nRows=image.rows;
	int nCols=image.cols*3;
	
	Mat dst;
	GaussianBlur(image,dst,Size(25,25),0,0);
		
	uchar *blurredRow,*extractedRow,*dstRow;
	for(int i=0;i<nRows;i++){
		extractedRow = maskedImg.ptr<uchar>(i);
		dstRow = dst.ptr<uchar>(i);
		
		for(int j=0;j<nCols;j+=3){
			if( extractedRow[j]!=0 && extractedRow[j+1]!=0 && extractedRow[j+2]!=0 ){
				dstRow[j]=extractedRow[j];
				dstRow[j+1]=extractedRow[j+1];
				dstRow[j+2]=extractedRow[j+2];
			}
		}
	}
	
	return dst;
}



Mat findHandContours(Mat& src){

	Mat src_gray;
	cvtColor( src, src_gray, COLOR_BGR2GRAY );
	
	RNG rng(12345);
	
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	/// Find contours
	findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Draw contours
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	 {
	   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	   drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	 }
	 
	 
   	/*vector<vector<Point> >hull( contours.size() );
	for( size_t i = 0; i < contours.size(); i++ )
	{   convexHull( Mat(contours[i]), hull[i], false ); }
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( size_t i = 0; i< contours.size(); i++ )
	{
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	drawContours( drawing, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	drawContours( drawing, hull, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	}*/

  	return drawing;
}


/* To remove any trackbars, comment out those parts */
void prepareTrackbarsNWindows(){
	namedWindow("Original Image",WINDOW_AUTOSIZE);
	namedWindow("HSV + BGR Mask",WINDOW_AUTOSIZE);
	namedWindow("HSV Mask",WINDOW_NORMAL);
	namedWindow("Masked Image",WINDOW_AUTOSIZE);
	namedWindow("Final Image",WINDOW_AUTOSIZE);
	namedWindow("Contours", WINDOW_AUTOSIZE );
	
	
	namedWindow("Blurring, Contouring Controllers",WINDOW_AUTOSIZE);
	createTrackbar("Blur kernel size","Blurring, Contouring Controllers",&kernSize,10);
	
	createTrackbar("MORPH OPEN size","Blurring, Contouring Controllers",&morphOpenKernSize,25);
	createTrackbar("MORPH CLOSE size","Blurring, Contouring Controllers",&morphCloseKernSize,25);
	createTrackbar("MORPH OPEN No Of Iterations size","Blurring, Contouring Controllers",&morphCloseNoOfIterations,25);
	
	createTrackbar("Contours Threshold","Blurring, Contouring Controllers",&thresh,255);
}
