#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;


Mat extractSkinColorRange(Mat& srcBGR,Mat& srcHSV,Mat& srcYCrCb);
bool isInSkinRangeBGR(const u_char& B,const u_char& G,const u_char& R);
bool isInSkinRangeHSV(const u_char& H,const u_char& S,const u_char& V);
bool isInSkinRangeYCrCb(const u_char& Y, const u_char& Cr, const u_char& Cb);
Mat findHandContours(Mat& src);
Mat combineExtractedWithMain(Mat& maskedImg,Mat& image);

int YMax=0,YMin=255,CrMax=0,CrMin=255,CbMax=0,CbMin=255;

int lH=0,hH=20,lS=20,hS=154,lV=50,hV=255,kernSize=2;
int Rlow=60,Glow=40,Blow=20,gap=15,Rhigh=220,Ghigh=210,Bhigh=170;
//OG Values
//int Rlow=95,Glow=40,Blow=20,gap=15,Rhigh=220,Ghigh=210,Bhigh=170;

int morphOpenKernSize=2,morphCloseKernSize=2;
int morphCloseNoOfIterations=3;

int thresh=100;


int main(int argc, char** argv){
	/*if(argc!=2){
		cout<<"No image argument specified";
		return -1;
	}
	
	Mat image = imread(argv[1],1);
	*/
	
	VideoCapture cap(0);
	
	if(!cap.isOpened()){
		return -1;
	}
	
	namedWindow("Original Image",WINDOW_AUTOSIZE);
	namedWindow("Controllers",WINDOW_AUTOSIZE);
	namedWindow("HSV + BGR Mask",WINDOW_AUTOSIZE);
	namedWindow("HSV Mask",WINDOW_NORMAL);
	namedWindow("Masked Image",WINDOW_AUTOSIZE);
	namedWindow("Final Image",WINDOW_AUTOSIZE);
	namedWindow( "Contours", WINDOW_AUTOSIZE );
	
	double maxTimeTaken=0,minTimeTaken=10000;
	
	createTrackbar("Low Hue","Controllers",&lH,180);
	createTrackbar("High Hue","Controllers",&hH,180);
	createTrackbar("Low Saturation","Controllers",&lS,255);
	createTrackbar("High Saturation","Controllers",&hS,255);
	createTrackbar("Low Value","Controllers",&lV,255);
	createTrackbar("High Value","Controllers",&hV,255);
	
	createTrackbar("Blur kernel size","Controllers",&kernSize,10);
	
	createTrackbar("I - R low","Controllers",&Rlow,255);
	createTrackbar("I - G low","Controllers",&Glow,255);
	createTrackbar("I - B low","Controllers",&Blow,255);
	createTrackbar("I - gap"  ,"Controllers",&gap,255);
	createTrackbar("II - R high","Controllers",&Rhigh,255);
	createTrackbar("II - G high","Controllers",&Ghigh,255);
	createTrackbar("II - B high","Controllers",&Bhigh,255);
	
	createTrackbar("MORPH OPEN size","Controllers",&morphOpenKernSize,25);
	createTrackbar("MORPH CLOSE size","Controllers",&morphCloseKernSize,25);
	createTrackbar("MORPH OPEN No Of Iterations size","Controllers",&morphCloseNoOfIterations,25);
	
	createTrackbar("Contours Threshold","Controllers",&thresh,255);
	
	while(true){
		Mat image;
		cap>>image;
		
		double startTime=(double)getTickCount();
		if(!image.data) continue;
		
		/* All processing functions go after this point */
		
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
		
		/* All processing functions come before this point */
		
		if(waitKey(20)=='q') break;
		
		double timeTaken=(getTickCount()-(double)startTime)/getTickFrequency();
		maxTimeTaken=timeTaken>maxTimeTaken?timeTaken:maxTimeTaken;
		minTimeTaken=timeTaken<minTimeTaken?timeTaken:minTimeTaken;
	}
	
	cout<<"Maximum time taken by one frame processing is "<<maxTimeTaken<<"s"<<endl;
	cout<<"Minimum time taken by one frame processing is "<<minTimeTaken<<"s"<<endl;
	cout<<YMax<<" "<<YMin<<" "<<CrMax<<" "<<CrMin<<" "<<CbMax<<" "<<CbMin<<endl;
	
	return 0;
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

Mat extractSkinColorRange(Mat& srcBGR,Mat& srcHSV,Mat& srcYCrCb){
	int nRows=srcBGR.rows;
	int nCols=srcBGR.cols*3;
	
	Mat dst(nRows,srcBGR.cols,CV_8UC1,Scalar(0));
		
	uchar *bgrRow,*hsvRow,*YCrCbRow,*dstRow;
	for(int i=0;i<nRows;i++){
		bgrRow = srcBGR.ptr<uchar>(i);
		hsvRow = srcHSV.ptr<uchar>(i);
		YCrCbRow = srcYCrCb.ptr<uchar>(i);
		dstRow = dst.ptr<uchar>(i);
		
		for(int j=0;j<nCols;j+=3){
			if( isInSkinRangeBGR(bgrRow[j],bgrRow[j+1],bgrRow[j+2])
				/*&& isInSkinRangeYCrCb(YCrCbRow[j],YCrCbRow[j+1],YCrCbRow[j+2])*/
				&& isInSkinRangeHSV(hsvRow[j],hsvRow[j+1],hsvRow[j+2]) ){
				dstRow[j/3]=255;
			}
		}
	}
	
	return dst;
}


bool isInSkinRangeYCrCb(const u_char& Y, const u_char& Cr, const u_char& Cb){
	//return Y>80 && Cb>85 && Cb<135 && Cr>135 && Cr<180;
	//cout<<Y<<" "<<Cr<<" "<<Cb<<endl;
	YMax=Y>YMax?Y:YMax;
	CrMax=Cr>CrMax?Cr:CrMax;
	CbMax=Cb>CbMax?Cb:CbMax;
	
	YMin=Y<YMin?Y:YMin;
	CrMin=Cr<CrMin?Cr:CrMin;
	CbMin=Cb<CbMin?Cb:CbMin;
	
	//return 1;
	bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
}


bool isInSkinRangeBGR(const u_char& B,const u_char& G,const u_char& R){
	/*
	 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
	 * Released to public domain under terms of the BSD Simplified license.
	 *
	 * Redistribution and use in source and binary forms, with or without
	 * modification, are permitted provided that the following conditions are met:
	 *   * Redistributions of source code must retain the above copyright
	 *     notice, this list of conditions and the following disclaimer.
	 *   * Redistributions in binary form must reproduce the above copyright
	 *     notice, this list of conditions and the following disclaimer in the
	 *     documentation and/or other materials provided with the distribution.
	 *   * Neither the name of the organization nor the names of its contributors
	 *     may be used to endorse or promote products derived from this software
	 *     without specific prior written permission.
	 *
	 *   See <http:www.opensource.org/licenses/bsd-license>
	 */
	 	
	bool e1 = (R>Rlow) && (G>Glow) && (B>Blow) && ((max(R,max(G,B)) - min(R, min(G,B)))>gap) && (abs(R-G)>gap) && (R>G) && (R>B);
    bool e2 = (R>Rhigh) && (G>Ghigh) && (B>Bhigh) && (abs(R-G)<=gap) && (R>B) && (G>B);
/*    bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);*/
    return (e1||e2);
}

bool isInSkinRangeHSV(const u_char& H,const u_char& S,const u_char& V){
	return ((H<hH) || (H > 155)) && S>=lS && S<=hS && V>=lV && V<=hV;
}
