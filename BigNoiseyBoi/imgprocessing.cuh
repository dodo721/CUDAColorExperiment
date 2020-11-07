#ifndef IMGPROCESSING_CUH
#define IMGPROCESSING_CUH

#include <stdint.h>

void LinearGenImageGPU(colour* imgData, unsigned int imgDataLength, colour* exclVec, size_t exclLength, bool verbose);
void LinearGenImageCPU(colour* imgData, unsigned int imgDataLength);
void InitialiseCUDA();
bool ValidateImage(colour* imgData, unsigned int imgDataLength);

#endif