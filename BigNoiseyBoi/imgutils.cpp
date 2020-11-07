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