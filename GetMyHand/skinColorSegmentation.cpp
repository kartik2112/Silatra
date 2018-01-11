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
 
/**
* This file contains the color extraction functions
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "trackBarHandling.hpp"

#include <iostream>

using namespace std;
using namespace cv;


Mat extractSkinColorRange(Mat& srcBGR,Mat& srcHSV,Mat& srcYCrCb);
bool isInSkinRangeBGR(const u_char& B,const u_char& G,const u_char& R);
bool isInSkinRangeHSV(const u_char& H,const u_char& S,const u_char& V);
bool isInSkinRangeYCrCb(const u_char& Y, const u_char& Cr, const u_char& Cb);



// int YMax=0,YMin=255,CrMax=0,CrMin=255,CbMax=0,CbMin=255;
int YMax=255,YMin=0,CrMax=180,CrMin=135,CbMax=130,CbMin=60;
// int lH=0,hH=20,lS=20,hS=154,lV=50,hV=255;
// int Rlow=60,Glow=40,Blow=20,gap=15,Rhigh=220,Ghigh=210,Bhigh=170;
int lH=0,hH=20,lS=8,hS=154,lV=50,hV=255;
int Rlow=60,Glow=40,Blow=20,gap=9,Rhigh=220,Ghigh=210,Bhigh=170;



Mat extractSkinColorRange(Mat& srcBGR,Mat& srcHSV,Mat& srcYCrCb){
	displaySkinColorDetectionTrackbarsIfNeeded();
	int nRows=srcBGR.rows;
	int nCols=srcBGR.cols*3;
	
	// static Mat dsts[3];
	// for(int i=0;i<3;i++){
	// 	dsts[i] = Mat(nRows,srcBGR.cols,CV_8UC1,Scalar(0));
	// }

	Mat dst(nRows,srcBGR.cols,CV_8UC1,Scalar(0));
		
	uchar *bgrRow, *hsvRow, *YCrCbRow, *dstRow;
	uchar *dstBGRRow, *dstHSVRow, *dstYCrCbRow;
	for(int i=0;i<nRows;i++){
		bgrRow = srcBGR.ptr<uchar>(i);
		hsvRow = srcHSV.ptr<uchar>(i);
		YCrCbRow = srcYCrCb.ptr<uchar>(i);
		dstRow = dst.ptr<uchar>(i);

		// dstBGRRow = dsts[0].ptr<uchar>(i);
		// dstHSVRow = dsts[1].ptr<uchar>(i);
		// dstYCrCbRow = dsts[2].ptr<uchar>(i);
		
		for(int j=0;j<nCols;j+=3){
			if( /* isInSkinRangeBGR(bgrRow[j],bgrRow[j+1],bgrRow[j+2])*/
				isInSkinRangeYCrCb(YCrCbRow[j],YCrCbRow[j+1],YCrCbRow[j+2])
				/* && isInSkinRangeHSV(hsvRow[j],hsvRow[j+1],hsvRow[j+2])*/ ){
				dstRow[j/3]=255;
			}
		}

		// for(int j=0;j<nCols;j+=3){
		// 	if( isInSkinRangeBGR(bgrRow[j],bgrRow[j+1],bgrRow[j+2])){
		// 		dstBGRRow[j/3]=255;
		// 	}
		// 	if( isInSkinRangeHSV(hsvRow[j],hsvRow[j+1],hsvRow[j+2]) ){
		// 		dstHSVRow[j/3]=255;
		// 	}
		// 	if( isInSkinRangeYCrCb(YCrCbRow[j],YCrCbRow[j+1],YCrCbRow[j+2]) ){
		// 		dstYCrCbRow[j/3]=255;
		// 	}
		// }
	}
	
	// return dsts;

	return dst;
}


bool isInSkinRangeYCrCb(const u_char& Y, const u_char& Cr, const u_char& Cb){
	return Y>YMin && Y<YMax && Cb>CbMin && Cb<CbMax && Cr>CrMin && Cr<CrMax;
	//cout<<Y<<" "<<Cr<<" "<<Cb<<endl;
	// CrCb low  Night 135,140

	// YMax=Y>YMax?Y:YMax;
	// CrMax=Cr>CrMax?Cr:CrMax;
	// CbMax=Cb>CbMax?Cb:CbMax;
	
	// YMin=Y<YMin?Y:YMin;
	// CrMin=Cr<CrMin?Cr:CrMin;
	// CbMin=Cb<CbMin?Cb:CbMin;
	
	// //return 1;
	// bool e3 = Cr <= 1.5862*Cb+20;
    // bool e4 = Cr >= 0.3448*Cb+76.2069;
    // bool e5 = Cr >= -4.5652*Cb+234.5652;
    // bool e6 = Cr <= -1.15*Cb+301.75;
    // bool e7 = Cr <= -2.2857*Cb+432.85;
    // return e3 && e4 && e5 && e6 && e7;
}


bool isInSkinRangeBGR(const u_char& B,const u_char& G,const u_char& R){	 	
	bool e1 = (R>Rlow) && (G>Glow) && (B>Blow) && ((max(R,max(G,B)) - min(R, min(G,B)))>gap) && (abs(R-G)>gap) && (R>G) && (R>B);
    bool e2 = (R>Rhigh) && (G>Ghigh) && (B>Bhigh) && (abs(R-G)<=gap) && (R>B) && (G>B);
/*    bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);*/
    return (e1||e2);
}

bool isInSkinRangeHSV(const u_char& H,const u_char& S,const u_char& V){
	return ((H<hH) || (H > 155)) && S>=lS && S<=hS && V>=lV && V<=hV;
}
