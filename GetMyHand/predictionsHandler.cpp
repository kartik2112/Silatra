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

deque<long long> predictions;
int maxQueueSize = 15;
int noOfSigns = 256;
int minModality = maxQueueSize/2;

void addPredictionToQueue(long long predictedSign){
    if(predictions.size()==maxQueueSize){
        predictions.pop_front();
    }
    predictions.push_back(predictedSign);
}

long long predictSign(){

    long long modePrediction = -1;
    int countModality = minModality;

    if(predictions.size()==maxQueueSize){
        int countPredictions[noOfSigns];

        for(int i = 0 ; i < noOfSigns ; i++){
            countPredictions[i] = 0;
        }

        for( deque<long long>::iterator it = predictions.begin() ; it != predictions.end() ; it++ ){
            countPredictions[*it]++;
        }


        for(int i = 0 ; i < noOfSigns ; i++){
            if(countPredictions[i]>countModality){
                modePrediction = i;
                countModality = countPredictions[i];
            }
        }        
    }
    
    displaySignOnImage(modePrediction);

    return modePrediction;
}

void displaySignOnImage(long long predictSign){
    Mat signImage = Mat::zeros(200,200,CV_8UC3);
    // cout<<predictSign<<endl;
    string dispSign = "--";
    if(predictSign!=-1){
        char c12 = (char)predictSign;
        dispSign = string(1,c12);
        // dispSign = to_string(predictSign);
        // cout<<(char)predictSign<<endl;
    }
    putText(signImage,dispSign,Point(75,100),FONT_HERSHEY_SIMPLEX,2,Scalar::all(255),3,8);

    imshow("Prediction",signImage);
}