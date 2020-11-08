#ifndef IMGUTILS_HPP
#define IMGUTILS_HPP

void showImg(cv::Mat* img, std::string windowName);
void showImg(cv::Mat* img, int width, int height, std::string windowName);
bool SortByColour(colour px1[3], colour px2[3]);

#endif