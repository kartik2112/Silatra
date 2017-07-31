/**
* This file should contain calls to the main functions in each phase only
* The functions specific to each phase must be defined in its own file
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "GetMyContours/getMyContours.hpp"

#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

void processFrame(Mat& image);

int main(int argc, char** argv){

	double maxTimeTaken=0,minTimeTaken=10000;	
	
	if(argc==2){
		Mat image = imread(argv[1],1);
		
		double startTime=(double)getTickCount();
		
		processFrame(image);
		
		waitKey(0);
	
		double timeTaken=(getTickCount()-(double)startTime)/getTickFrequency();
		maxTimeTaken=timeTaken>maxTimeTaken?timeTaken:maxTimeTaken;
		minTimeTaken=timeTaken<minTimeTaken?timeTaken:minTimeTaken;
		
	}
	else{
		
		VideoCapture cap(0);
	
		if(!cap.isOpened()){
			return -1;
		}	
		
	
		while(true){
			Mat image;
			cap>>image;
		
			double startTime=(double)getTickCount();
			
			if(!image.data) continue;
		
			processFrame(image);
		
			if(waitKey(20)=='q') break;
		
			double timeTaken=(getTickCount()-(double)startTime)/getTickFrequency();
			maxTimeTaken=timeTaken>maxTimeTaken?timeTaken:maxTimeTaken;
			minTimeTaken=timeTaken<minTimeTaken?timeTaken:minTimeTaken;
		}
	
		
		//cout<<YMax<<" "<<YMin<<" "<<CrMax<<" "<<CrMin<<" "<<CbMax<<" "<<CbMin<<endl;
	
	}
	
	
	cout<<"Maximum time taken by one frame processing is "<<maxTimeTaken<<"s"<<endl;
	cout<<"Minimum time taken by one frame processing is "<<minTimeTaken<<"s"<<endl;
	
		
	
	return 0;
}

void processFrame(Mat& image){
	/* All processing functions go after this point */
		
	getMyContours(image);  //Defined in GetMyContours/getMyContours.cpp

	/* All processing functions come before this point */
}
