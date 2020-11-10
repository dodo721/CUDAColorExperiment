#ifndef IMGPROCESSING_CUH
#define IMGPROCESSING_CUH

#include <stdint.h>

void LinearGenImageGPU(colour* imgData, unsigned int imgDataLength, ColourEntry* exclusionIndex, bool verbose);
void LinearGenImageCPU(colour* imgData, unsigned int imgDataLength, ColourEntry* exclusionIndex, bool verbose);
void InitialiseCUDA();
bool ValidateImage(colour* imgData, unsigned int imgDataLength);

#endif