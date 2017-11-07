/**
* This file contains the code that will dilate, erode the regions extracted based on color
*/

#include <python3.5/Python.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "skinColorSegmentation.hpp"
#include "trackBarHandling.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

#include <experimental/filesystem>

#define OVERALL 0
#define SKIN_COLOR_EXTRACTION 1
#define MORPHOLOGY_OPERATIONS 2
#define MODIFIED_IMAGE_GENERATION 3
#define HAND_CONTOURS_GENERATION 4
#define CONTOURS_PRE_PROCESSING 5
#define CONTOURS_IMPROVEMENT 6
#define CONTOURS_POST_PROCESSING 7
#define CONTOUR_CLASSIFICATION_IN_PY 8

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

Mat getMyHand(Mat& image);
Mat findHandContours(Mat& src);
Mat combineExtractedWithMain(Mat& maskedImg,Mat& image);
void prepareWindows();
void connectContours(vector<vector<Point> > &contours);
void reduceClusterPoints(vector< vector< Point > > &contours);
void findClassUsingPythonModels( vector<float> &distVector );




int morphOpenKernSize=2,morphCloseKernSize=2;
int morphCloseNoOfIterations=3;

int kernSize=2;
int thresh=100;
int contourDistThreshold = 30;
double startTime;

extern int lH,lS,lV,hH,hS,hV;
extern string subDirName;

extern vector<double> frameStepsTimes;

extern char** args;
extern int args_c;


/*
This is the main entry point function of this file
*/
Mat getMyHand(Mat& imageOG){

	displayHandDetectionTrackbarsIfNeeded(imageOG);
	
	startTime=(double)getTickCount();  //---Timing related part
	
	imshow("Original Image",imageOG);
	Mat image,imageHSV,imageYCrCb;
	
	/*
	cvtColor(image,imageYCrCb,CV_BGR2YCrCb);
	Mat HEd(image.rows,image.cols,CV_8UC1,Scalar(0));	
	int fromToArr1[] = {0,0};
	mixChannels(imageYCrCb,HEd,fromToArr1,1);
	equalizeHist( HEd, HEd );
	mixChannels(HEd,imageYCrCb,fromToArr1,1);	
	cvtColor(imageYCrCb,image,CV_YCrCb2BGR);
	imshow("HEd Image",image);
	*/
	
	//blur(image,image,Size(kernSize,kernSize),Point(-1,-1));
	GaussianBlur(imageOG,image,Size(2*kernSize+1,2*kernSize+1),0,0);
/**/	imshow("Gaussian Blurred Image",image); 
	//medianBlur(image,image,2*kernSize+1);
	
	/* Convert BGR Image into HSV, YCrCb Images */
	cvtColor(image,imageHSV,CV_BGR2HSV);
	cvtColor(image,imageYCrCb,CV_BGR2YCrCb);

	Mat dstHSV;
	inRange(imageHSV,Scalar(lH,lS,lV),Scalar(hH,hS,hV),dstHSV);
	
	Mat dst=extractSkinColorRange(image,imageHSV,imageYCrCb);
/**/	imshow("Skin Color Range Pixels Extracted Image (using HSV, BGR ranges)",dst); 
	
	
	frameStepsTimes[ SKIN_COLOR_EXTRACTION ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	startTime=(double)getTickCount();  //---Timing related part
	
	Mat morphOpenElement = getStructuringElement(MORPH_CROSS,Size(morphOpenKernSize*2+1,morphOpenKernSize*2+1),Point(morphOpenKernSize,morphOpenKernSize));
	Mat morphCloseElement = getStructuringElement(MORPH_CROSS,Size(morphCloseKernSize*2+1,morphCloseKernSize*2+1),Point(morphCloseKernSize,morphCloseKernSize));
	Mat dstEroded;
	
// 	erode(dst,dstEroded,morphOpenElement);
// /**/	imshow("Round 1 - Eroded Segment - MorphologyEx - MORPH_OPEN",dstEroded);
// 	dilate(dstEroded,dstEroded,morphOpenElement);
// /**/	imshow("Round 1 - Dilated Segment - MorphologyEx - MORPH_OPEN",dstEroded);
	
// 	dilate(dstEroded,dstEroded,morphOpenElement);
// /**/	imshow("Round 2 - Dilated Segment - MorphologyEx - MORPH_CLOSE",dstEroded);
// 	erode(dstEroded,dstEroded,morphOpenElement);
// /**/	imshow("Round 2 - Eroded Segment - MorphologyEx - MORPH_CLOSE",dstEroded);
	
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
/**/	imshow("Round 3 - Dilated Segment - to expand segmented area",dstEroded);
	
	
	//cout<<dst.type()<<" "<<image.type()<<endl;
	
	
	Mat morphCloseElement1 = getStructuringElement(MORPH_ELLIPSE,Size(15,15),Point(7,7));
	/* AND this eroded mask with HSV */
	//bitwise_and(dstEroded,dstHSV,dstEroded);
	
	morphologyEx(dstEroded,dstEroded,MORPH_CLOSE,morphCloseElement1);
	
	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE,Size(8,8),Point(4,4));
	/* This will enlarge white areas */
	dilate(dstEroded,dstEroded,dilateElement,Point(-1,-1),morphCloseNoOfIterations);
	//dilate(dstEroded,dstEroded,dilateElement,Point(-1,-1),morphCloseNoOfIterations);
/**/	imshow("Round 4,5 - After morphologyEx(MORPH_CLOSE) and dilate segment",dstEroded);

	frameStepsTimes[ MORPHOLOGY_OPERATIONS ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	startTime=(double)getTickCount();  //---Timing related part

	Mat maskedImg;
	cvtColor(dstEroded,dstEroded,CV_GRAY2BGR);
	bitwise_and(dstEroded,imageOG,maskedImg);
	
	Mat finImg=combineExtractedWithMain(maskedImg,image);

	frameStepsTimes[ MODIFIED_IMAGE_GENERATION ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	startTime=(double)getTickCount();  //---Timing related part
	
	Mat contouredImg=findHandContours(finImg);

	frameStepsTimes[ HAND_CONTOURS_GENERATION ] = (getTickCount()-(double)startTime)/getTickFrequency();   //---Timing related part
	startTime=(double)getTickCount();  //---Timing related part
	
	/// Show in a window  
	imshow("Contours", contouredImg );	
	//imwrite("./ContourImages/img.png",contouredImg);
	imshow("Morphed Mask",dstEroded);
	imshow("Masked Image",maskedImg);
	imshow("Final Image",finImg);
	imshow("HSV + BGR Mask",dst);
	imshow("HSV Mask",dstHSV);



	// Py_SetProgramName();
	// Py_Initialize();
	// PyRun_SimpleString("print '\nThis is first successful Python Statement being run from C++' ");
	// Py_Finalize();
	



	return contouredImg;	
}


Mat combineExtractedWithMain(Mat& maskedImg,Mat& image){
	int nRows=image.rows;
	int nCols=image.cols*3;
	
	Mat dst;
	GaussianBlur(image,dst,Size(25,25),0,0);
/**/	imshow("Gaussian Blur on BG",dst);
		
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

	double startTime1 = (double)getTickCount();  //---Timing related part


	Mat src_gray;
	cvtColor( src, src_gray, COLOR_BGR2GRAY );
	
	RNG rng(12345);
	
	Mat canny_output;
	vector<vector<Point> > contours,contours1;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	
	Mat morphCloseElement = getStructuringElement(MORPH_ELLIPSE,Size(5*2+1,5*2+1),Point(5,5));
	//morphologyEx(canny_output,canny_output,MORPH_CLOSE,morphCloseElement);
	
	imshow("Canny",canny_output);
	/// Find contours
	findContours( canny_output, contours1, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	contours.resize(contours1.size());
	
	for( size_t i = 0; i< contours.size(); i++ )
	{	
		cout<<"Contour "<<(i+1)<<" size: "<<contours1[i].size()<<":"<<endl;
		// for(auto p:contours1[i]){
		// 	cout<<"("<<p.x<<","<<p.y<<")"<<", ";
		// }
		// cout<<endl<<endl;
	}
	
	/// Draw contours
	Mat drawingOGContours = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( int i = 0; i< contours1.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawingOGContours, contours1, i, color, 1, 8, hierarchy, 0, Point() );
		//cout<<contours[i].size()<<endl;
	}
	imshow("OG Contours",drawingOGContours);
	
	
	
	
	for( size_t i = 0; i< contours1.size(); i++ )
	{		
		approxPolyDP(Mat(contours1[i]),contours[i],3,true);
	}




	frameStepsTimes[ CONTOURS_PRE_PROCESSING ] = (getTickCount()-(double)startTime1)/getTickFrequency();   //---Timing related part
	startTime1=(double)getTickCount();  //---Timing related part


	
	connectContours( contours );
	
	reduceClusterPoints( contours );


	frameStepsTimes[ CONTOURS_IMPROVEMENT ] = (getTickCount()-(double)startTime1)/getTickFrequency();   //---Timing related part
	startTime1=(double)getTickCount();  //---Timing related part


	
   	vector<vector<Point> >hull( contours.size() );
	for( size_t i = 0; i < contours.size(); i++ )
	{   convexHull( Mat(contours[i]), hull[i], false ); }
	
	
	/// Get the moments
	vector<Moments> mu(hull.size() );
	for( int i = 0; i < hull.size(); i++ )
	{ mu[i] = moments( hull[i], false ); }

	///  Get the mass centers:
	vector<Point2f> mc( hull.size() );
	for( int i = 0; i < hull.size(); i++ )
	{ mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

	 
	 
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( size_t i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		circle( drawing, mc[i], 4, color, -1, 8, 0 );
		cout<<"Contour "<<(i+1)<<" size: "<<contours[i].size()<<":"<<endl;
		circle( drawing, contours[i][0], 4, Scalar(255,0,0), -1, 8, 0 );
		circle( drawing, contours[i][10], 4, Scalar(255,0,0), -1, 8, 0 );
		// for(auto p:contours[i]){
		// 	cout<<"("<<p.x<<","<<p.y<<")"<<", ";
		// }
		// cout<<endl<<endl;
	}
	
	double maxArea = 0;
	int indMaxArea=-1;
	for( int i = 0; i < hull.size(); i++ )
	{
		double area12 = contourArea(hull[i]);
		if(area12>maxArea){
			indMaxArea = i;
			maxArea = area12;
		}
	}
	
	
	vector<int> convHull;
	convexHull( Mat(contours[indMaxArea]), convHull, false );   // Reason to keep int instead of Point for convexHull vector:
																//   https://stackoverflow.com/a/20552514/5370202
	vector<Vec4i> convD;
	if(contours[indMaxArea].size()>2 && convHull.size()>2)
		convexityDefects(contours[indMaxArea],convHull,convD);
	int contourPoints=0;
	
	for(int i=0;i<convD.size();i++){
		int start,end,far;
		double farDist;
		
		start = convD[i][0]; end = convD[i][1];
		far = convD[i][2]; farDist = convD[i][3];
		// cout<<start<<","<<end<<","<<far<<","<<farDist/256.0<<endl;
		if(farDist/256.0 > 70){
			contourPoints++;
		}
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		//circle( drawing, contours[indMaxArea][start], 4, color, -1, 8, 0 );
		//circle( drawing, contours[indMaxArea][end], 4, color, -1, 8, 0 );
		//circle( drawing, contours[indMaxArea][far], 4, color, -1, 8, 0 );
	}
	
	// cout<<contourPoints<<" Convex Defects Detected"<<endl;
	
	
		
	ofstream csvFile;
	csvFile.open(subDirName+"/data.csv",std::ios_base::app);
	vector<float> distVector(contours[indMaxArea].size());
	float maxDist = 0;
	for(int i=0;i<contours[indMaxArea].size();i++){
		float dist=sqrt( (contours[indMaxArea][i].x-mc[indMaxArea].x)*(contours[indMaxArea][i].x-mc[indMaxArea].x) + (contours[indMaxArea][i].y-mc[indMaxArea].y)*(contours[indMaxArea][i].y-mc[indMaxArea].y) );
		distVector[i]=dist;
		if(dist>maxDist){
			maxDist = dist;
		}
	}
	
	for(int i=0;i<contours[indMaxArea].size();i++){
		distVector[i] = (distVector[i]/maxDist*10);
		csvFile << distVector[i];
		if(i!=contours[indMaxArea].size()-1) csvFile << ", ";
	}
	csvFile << "\n";
	csvFile.close();



	frameStepsTimes[ CONTOURS_POST_PROCESSING ] = (getTickCount()-(double)startTime1)/getTickFrequency();   //---Timing related part
	startTime1=(double)getTickCount();  //---Timing related part



	if( (args_c==3 && ( strcmp(args[1],"-img")==0 || strcmp(args[1],"-AllImgs")==0 ) ) || waitKey(30)=='m' )
		findClassUsingPythonModels(distVector);



	frameStepsTimes[ CONTOUR_CLASSIFICATION_IN_PY ] = (getTickCount()-(double)startTime1)/getTickFrequency();   //---Timing related part


	contours.clear();
	contours.shrink_to_fit();

	contours1.clear();
	contours1.shrink_to_fit();

  	return drawing;
}


void prepareWindows(){
	//namedWindow("Original Image",WINDOW_NORMAL);
	//namedWindow("HSV + BGR Mask",WINDOW_NORMAL);
	//namedWindow("HSV Mask",WINDOW_NORMAL);
	namedWindow("Masked Image",WINDOW_NORMAL);
	namedWindow("Final Image",WINDOW_NORMAL);
	namedWindow("Contours", WINDOW_NORMAL );
}


/* 
This function needs huge optimization!!!!! 
Can be optimized using RTrees
*/
void connectContours(vector<vector<Point> > &contours){
	
	int ctr1=-1,ctr2=-1,ctr1PtIndex,ctr2PtIndex;
	long long contourDistThresholdL=contourDistThreshold*contourDistThreshold;
	long long minDist=contourDistThresholdL*10;
	
	long long countComparisons = 0;
	while(true){
		bool canMerge = false;
		minDist=contourDistThresholdL*10;
		for(int i=0;i<contours.size()-1;i++){
			for(int j=i+1;j<contours.size();j++){
				for(int iPt=0;iPt<contours[i].size();iPt++){
					for(int jPt=0;jPt<contours[j].size();jPt++){
						long long tempDist = (contours[i][iPt].x-contours[j][jPt].x)*(contours[i][iPt].x-contours[j][jPt].x) + (contours[i][iPt].y-contours[j][jPt].y)*(contours[i][iPt].y-contours[j][jPt].y);
						countComparisons++;
						if( tempDist<=contourDistThresholdL && tempDist < minDist)
						{
							minDist = tempDist;
							ctr1 = i; ctr2 = j; ctr1PtIndex = iPt; ctr2PtIndex = jPt;
							canMerge = true;
							
						}
					}
				}
			}
		}
		
		if(!canMerge){
			break;
		}
		else{
			cout<<ctr1<<" merged with "<<ctr2<<endl;
			//contours[ctr1].erase(contours[ctr1].begin()+ctr1PtIndex+1,contours[ctr1].end());
			//contours[ctr2].erase(contours[ctr2].begin(),contours[ctr2].begin()+ctr2PtIndex);
			//contours[ctr1].insert(contours[ctr1].begin(),contours[ctr2].begin(),contours[ctr2].end());
			contours[ctr1].insert(contours[ctr1].begin()+ctr1PtIndex+1,contours[ctr2].begin()+ctr2PtIndex,contours[ctr2].end());
			int insertedSize = contours[ctr2].size()-ctr2PtIndex+1;
			contours[ctr1].insert(contours[ctr1].begin()+ctr1PtIndex+insertedSize,contours[ctr2].begin(),contours[ctr2].begin()+ctr2PtIndex+1);
			contours.erase(contours.begin()+ctr2);
		}
	}

	cout<<countComparisons<<" no of comparisons made!"<<endl;
	//contours[0].erase(contours[0].begin()+contours[0].size()/2,contours[0].end());
}


/*
Okayishly efficient but horribly innacurate for noisy intersecting contours
*/
void reduceClusterPoints(vector< vector< Point > > &contours){
	int maxI = -1, max = 0;
	for(int i = 0 ; i < contours.size() ; i++){
		cout<<contours[i].size()<<", ";
		if(contours[i].size()>max){
			max = contours[i].size();
			maxI = i;
		}
	}
	cout<<endl;
	
	long long minDist = 9223372036854775805;
	int minDistI = -1;
	for(int i = 7 ; i < contours[ maxI ].size() - 7; i++){
		long long tempDist = ( contours[maxI][i].x - contours[maxI][0].x ) * ( contours[maxI][i].x - contours[maxI][0].x ) + ( contours[maxI][i].y - contours[maxI][0].y ) * ( contours[maxI][i].y - contours[maxI][0].y );
		if(tempDist>=0 && tempDist<=minDist){
			minDist = tempDist;
			minDistI = i;
		}
	}
	
	int firstHalfEnd = minDistI / 2, secondHalfEnd = minDistI + (contours[maxI].size() - minDistI - 1) / 2;
	
	cout<<firstHalfEnd<<", "<<minDistI<<", "<<secondHalfEnd<<", "<<contours[maxI].size()-1<<endl;
	contours[maxI].insert(contours[maxI].begin(), contours[maxI].begin()+secondHalfEnd, contours[maxI].end());
	contours[maxI].erase(contours[maxI].begin()+firstHalfEnd+secondHalfEnd-minDistI,contours[maxI].end());
}


void findClassUsingPythonModels( vector<float> &distVector ){

	ofstream csvFile;
	fs::remove("./TestSampleDistancesData.csv");
	csvFile.open("./TestSampleDistancesData.csv",std::ios_base::app);

	for(int i=0;i<distVector.size();i++){
		csvFile << distVector[i];
		if(i!=distVector.size()-1) csvFile << ", ";
	}

	csvFile << "\n";
	csvFile.close();

	cout<<endl<<endl<<endl<<"Python Invocation from from C++ begins here"<<endl;


	/*
	References for Python interfacing: 
	Main code: https://docs.python.org/2/extending/embedding.html
	For adding target_link_libraries in CMakeLists.txt: https://stackoverflow.com/a/21548557/5370202
	*/

	// PyRun_SimpleString("print('Python Invocation from from C++ begins here')");
	// Py_SetProgramName();


	// Py_Initialize();   //Moved to main() of Silatra.cpp
	FILE* file = fopen("testThisSampleInPython.py","r");
	PyRun_SimpleFile(file,"testThisSampleInPython.py");
	// PyRun_SimpleFile(file,"LoadSavedModel.py");
	// Py_Finalize();   //Moved to main() of Silatra.cpp
	fclose(file);

	cout<<"Python Invocation ends here"<<endl<<endl<<endl;

}