#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;
using namespace std; 

int main(int argc, char** argv )
{
	clock_t clockStart=clock();
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
	cout<<(double)(clock()-clockStart)/CLOCKS_PER_SEC<<endl;

    waitKey(0);

    return 0;
}
