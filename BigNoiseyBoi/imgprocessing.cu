#include "defines.hpp"

#include <iostream>
#include <vector>
#include <math.h>
#include <ppl.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "colour_indexer.cuh"
#include "imgutils.hpp"
#include "imgprocessing.cuh"

using namespace concurrency;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KERNELS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Exclusion-checking algorithm
__global__
void LinearGenPixelColour(colour* imgData, unsigned int imgDataLength, ColourEntry* exclusionIndex) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    bool occupied = exclusionIndex[index].occupied;
    // Occupied is 0 or 1 therefore 1-occupied=inverse
    
    // Use the index if not occupied, use the index, otherwise the end of the image + the exclusion order
    int val = ((1 - occupied) * index) + (occupied * (imgDataLength + exclusionIndex[index].order));

    // Apply colour calculations
    if (threadIdx.y == 1)
        val /= 256;
    else if (threadIdx.y == 2)
        val /= 65536;
    
    imgData[(index * 3) + threadIdx.y] = val;

    // NO NEED FOR MODULO - BYTE OVERFLOW DOES IT FOR YOU
}

// Non-exclusion checking algorithm
__global__
void LinearGenPixelColour(colour* imgData) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    int val = index;
    if (threadIdx.y == 1)
        val /= 256;
    else if (threadIdx.y == 2)
        val /= 65536;
    imgData[(index * 3) + threadIdx.y] = val;

    // NO NEED FOR MODULO - BYTE OVERFLOW DOES IT FOR YOU
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LinearGenImageCPU(colour* imgData, unsigned int imgDataLength) {

    for (unsigned int i = 0; i < imgDataLength; i ++) {
        colour* pixel = imgData + (i * 3);
        int val = i;
        for (char j = 0; j < 3; j++) {
            // NO NEED FOR MODULO - BYTE OVERFLOW DOES IT FOR YOU
            *(pixel + j) = val;
            val /= 256;
        }
    }
}

void LinearGenImageCPU(colour* imgData, unsigned int imgDataLength, ColourEntry* exclusionIndex, bool verbose) {

    if (exclusionIndex == nullptr) {
        // Non-exclusion checking algorithm
        LinearGenImageCPU(imgData, imgDataLength);
    }
    else {
        // Exclusion checking algorithm
        for (unsigned int i = 0; i < imgDataLength; i++) {
            colour* pixel = imgData + (i * 3);
            bool occupied = exclusionIndex[i].occupied;
            int val = ((1 - occupied) * i) + (occupied * (imgDataLength + exclusionIndex[i].order));
            for (char j = 0; j < 3; j++) {
                // NO NEED FOR MODULO - BYTE OVERFLOW DOES IT FOR YOU
                *(pixel + j) = val;
                val /= 256;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LinearGenImageGPU(colour* imgData, unsigned int imgDataLength, ColourEntry* exclusionIndex, bool verbose) {

    colour* gpuImgData;
    cudaMalloc(&gpuImgData, imgDataLength * sizeof(colour));
    
    // MAX BLOCK SIZE:
    //  X: 512
    //  Y: 512
    //  Z: 64
    // TOTAL MUST NOT EXCEED 1024
    int threadsX = 256;
    int threadsY = 3;
    dim3 blockSize(threadsX, threadsY);
    int blockMag = threadsX * threadsY;
    int numBlocks = (imgDataLength + blockMag - 1) / blockMag;
    
    if (verbose)
        cout << "GPU Thread setup: " << numBlocks << " blocks, " << threadsX << "x" << threadsY << " block size, " << (numBlocks * blockMag) << " total threads" << endl;

    if (exclusionIndex != nullptr)
        LinearGenPixelColour <<<numBlocks, blockSize>>> (gpuImgData, imgDataLength, exclusionIndex);
    else
        LinearGenPixelColour <<<numBlocks, blockSize>>> (gpuImgData);
    cudaDeviceSynchronize();

    cudaMemcpy(imgData, gpuImgData, imgDataLength * sizeof(colour), cudaMemcpyDeviceToHost);
    cudaFree(gpuImgData);
}

void InitialiseCUDA() {
    char* initMem;
    cudaMalloc(&initMem, sizeof(char));
    cudaFree(initMem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VALIDATION
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO Include exclusions
bool ImageIsValid(colour* imgData, size_t imgDataLengthPixels) {
    size_t imgDataLength = imgDataLengthPixels * 3;

    // Optmisation: assign image to 2D array structure so it can be sorted
    // Sort array according to colour
    colour** sortedImgData = new colour*[imgDataLengthPixels];
    for (int i = 0; i < imgDataLengthPixels; i++) {
        sortedImgData[i] = new colour[3];
        sortedImgData[i][0] = imgData[i * 3];
        sortedImgData[i][1] = imgData[i * 3 + 1];
        sortedImgData[i][2] = imgData[i * 3 + 2];
    }
    sort(sortedImgData, sortedImgData + imgDataLengthPixels, SortByColour);

    /*
    int imgWidth = (int)sqrt(imgDataLengthPixels);
    for (int i = 0; i < imgDataLengthPixels; i++) {
        cout << "(" << (int)sortedImgData[i][0] << "," << (int)sortedImgData[i][1] << "," << (int)sortedImgData[i][2] << "), ";
        if (i % imgWidth == 0)
            cout << "]\n[ ";
    }
    cout << endl;
    */

    // As array is now sorted, any identical colours will be adjacent
    bool ret = true;
    parallel_for(size_t(0), imgDataLengthPixels, [&](size_t i) {
        if (i != 0 && ColourEquals(sortedImgData[i], sortedImgData[i - 1])) {
            ret = false;
            cout << "COLOUR COLLISION: " << (int)sortedImgData[i][0] << "," << (int)sortedImgData[i][1] << "," << (int)sortedImgData[i][2] << endl;
        }
        else if (i != imgDataLengthPixels - 1 && ColourEquals(sortedImgData[i], sortedImgData[i + 1])) {
            ret = false;
            cout << "COLOUR COLLISION: " << (int)sortedImgData[i][0] << "," << (int)sortedImgData[i][1] << "," << (int)sortedImgData[i][2] << endl;
        }
    });

    for (int i = 0; i < imgDataLengthPixels; i++) {
        delete[] sortedImgData[i];
    }
    delete[] sortedImgData;
    return ret;
}

bool ValidateImage(colour* imgData, unsigned int imgDataLengthPixels) {
    // Put in worker thread to keep program responding
    double t = (double)cv::getTickCount();
    bool valid = ImageIsValid(imgData, imgDataLengthPixels);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Image validated in " << t  << "s" << endl;
    return valid;
    //thread worker (ImageIsValid, imgData, imgDataLengthPixels);
    //worker.join();
}