/**
* This file should contain calls to the main functions in each phase only
* The functions specific to each phase must be defined in its own file
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "GetMyHand/handDetection.hpp"

#include <iostream>
#include <ctime>
#include <experimental/filesystem>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

void processFrame(Mat& image);

string subDirName;

int main(int argc, char** argv){

	double maxTimeTaken=0,minTimeTaken=10000;
	
	if(argc==3 && strcmp(argv[1],"-img")==0){	
		subDirName = string(argv[2]);
		subDirName = "./CCDC-Data/"+subDirName.substr(0,subDirName.find_last_of("/"));
		fs::create_directories(subDirName);
				
		
		Mat image = imread(argv[2],1);
		
		double startTime=(double)getTickCount();
		
		processFrame(image);		
	
		double timeTaken=(getTickCount()-(double)startTime)/getTickFrequency();
		maxTimeTaken=timeTaken>maxTimeTaken?timeTaken:maxTimeTaken;
		minTimeTaken=timeTaken<minTimeTaken?timeTaken:minTimeTaken;
		
		waitKey(0);
	}
	else if(argc==3 && strcmp(argv[1],"-AllImgs")==0){
		string subDirName1 = string(argv[2]);
		subDirName = "./CCDC-Data/"+subDirName1.substr(0,subDirName1.find_last_of("/"));
		fs::create_directories(subDirName);
		fs::remove(subDirName+"/data.csv");
		
		vector<string> files;
		for(auto &tempp1:fs::directory_iterator(subDirName1)){
			files.push_back(tempp1.path().string());
		}

		sort(files.begin(),files.end());

		for(int i=0;i<files.size();i++){
			cout<<"Processing "<<files[i]<<endl;
			Mat image = imread(files[i],1);
			
			double startTime=(double)getTickCount();
			
			processFrame(image);		
		
			double timeTaken=(getTickCount()-(double)startTime)/getTickFrequency();
			maxTimeTaken=timeTaken>maxTimeTaken?timeTaken:maxTimeTaken;
			minTimeTaken=timeTaken<minTimeTaken?timeTaken:minTimeTaken;
		}
	}
	else{
	
		string trainingImagesFolderPath;
		int imgNo=1;
		if(argc==3 && strcmp(argv[1],"-cap")==0){
			/*cout<<"Enter name of subdirectory for storing the training images: "<<endl;*/
			subDirName = string(argv[2]);
			//cin>>subDirName;
			trainingImagesFolderPath="./training-images/"+subDirName;
			fs::create_directories(trainingImagesFolderPath);
			fs::create_directories("./CCDC-Data/"+subDirName);
			for(auto &tempp1:fs::directory_iterator(trainingImagesFolderPath)){
				imgNo++;
			}
			//mkdir("./training-images/"+subDirName);
		}
		
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
		
			double timeTaken=(getTickCount()-(double)startTime)/getTickFrequency();
			maxTimeTaken=timeTaken>maxTimeTaken?timeTaken:maxTimeTaken;
			minTimeTaken=timeTaken<minTimeTaken?timeTaken:minTimeTaken;
			
			if(waitKey(20)=='q') break;
			if(argc==3 && waitKey(30)=='c'){
				imwrite(trainingImagesFolderPath+"/"+to_string(imgNo)+".png",image);
				imgNo++;
			}
		}
	
		
	
	}
	
	
	cout<<"Maximum time taken by one frame processing is "<<maxTimeTaken<<"s"<<endl;
	cout<<"Minimum time taken by one frame processing is "<<minTimeTaken<<"s"<<endl;
	
		
	
	return 0;
}

void processFrame(Mat& image){
	/* All processing functions go after this point */
		
	Mat contours = getMyHand(image);  //Defined in GetMyContours/getMyContours.cpp

	/* All processing functions come before this point */
}
