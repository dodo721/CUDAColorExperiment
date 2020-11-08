#include "defines.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include "imgutils.hpp"

using namespace std;
using namespace cv;

void showImg(Mat* img, string windowName)
{
	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, img->rows, img->cols);
	imshow(windowName, *img);
	waitKey(0);
}

void showImg(Mat* img, int width, int height, string windowName)
{
	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, width, height);
	imshow(windowName, *img);
}

bool SortByColour(colour px1[3], colour px2[3]) {
	if (px1[0] > px2[0]) return true;
	else if (px1[0] < px2[0]) return false;
	else if (px1[1] > px2[1]) return true;
	else if (px1[1] < px2[1]) return false;
	else return px1[2] > px2[2];
}