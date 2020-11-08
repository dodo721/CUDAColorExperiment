#include "defines.hpp"

#include <iostream>
#include <vector>
#include <math.h>
#include <ppl.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "colour_indexer.cuh"
#include "imgprocessing.cuh"

using namespace concurrency;
using namespace std;

__global__
void LinearGenPixelColour(colour* imgData, unsigned int imgDataLength, ColourEntry* exclusionIndex, size_t exclusionsLength, int* conflictingIndexes) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //imgData[colourIndex] = pixelIndex + (((threadIdx.y * 2) + 1) * blockIdx.x);
    int addition = (exclusionIndex[index].occupied * imgDataLength);
    int val = index + (exclusionIndex[index].occupied * imgDataLength);
    if (exclusionIndex[index].occupied) {
        printf("Index: %d, addition: (%d * %d = %d), val: %d\n", index, (int)exclusionIndex[index].occupied, imgDataLength, addition, val);
        printf("Colour: %d,%d,%d\n", val % 256, (val / 256) % 256, (val / 65536) % 256);
        printf("OG Colour: %d,%d,%d\n", index % 256, (index / 256) % 256, (index / 65536) % 256);
    }
    if (threadIdx.y == 1)
        val /= 256;
    else if (threadIdx.y == 2)
        val /= 65536;
    
    imgData[(index * 3) + threadIdx.y] = val;

    // NO NEED FOR MODULO - BYTE OVERFLOW DOES IT FOR YOU
}

__global__
void LinearGenPixelColour(colour* imgData, unsigned int imgDataLength) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //imgData[colourIndex] = pixelIndex + (((threadIdx.y * 2) + 1) * blockIdx.x);

    int val = index;
    if (threadIdx.y == 1)
        val /= 256;
    else if (threadIdx.y == 2)
        val /= 65536;
    imgData[(index * 3) + threadIdx.y] = val;

    // NO NEED FOR MODULO - BYTE OVERFLOW DOES IT FOR YOU

    /*int val = i;
    int temp;
    for (char j = 0; j < 3; j++) {
        // Use temp variable to allow divide and modulo in this order, to take advantage of x86 remainder function and reduce divisions by half
        temp = val / 256;
        *(pixel + j) = val % 256;
        val = temp;
    }*/
}

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

void LinearGenImageGPU(colour* imgData, unsigned int imgDataLength, colour* exclArr, size_t exclLength, bool verbose) {

    cout << "Indexing " << exclLength << " colours" << endl;
    ColourEntry* exclusions = initColourIndexer();
    if (exclArr != nullptr) {
        double t = cv::getTickCount();
        prepareExclusionList(exclusions, exclArr, exclLength);
        t = (cv::getTickCount() - t) / cv::getTickFrequency();
        if (verbose) cout << "Exclusions indexed in " << t << endl;
    }

    colour* gpuImgData;
    cudaMalloc(&gpuImgData, imgDataLength * sizeof(colour));

    int* excludedIndexes = new int[exclLength];
    
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

    if (exclArr != nullptr)
        LinearGenPixelColour <<<numBlocks, blockSize>>> (gpuImgData, imgDataLength / 3, exclusions, exclLength, excludedIndexes);
    else
        LinearGenPixelColour <<<numBlocks, blockSize>>> (gpuImgData, imgDataLength / 3);
    cudaDeviceSynchronize();

    cudaMemcpy(imgData, gpuImgData, imgDataLength * sizeof(colour), cudaMemcpyDeviceToHost);
    cudaFree(gpuImgData);
    cudaFree(exclusions);
}

void InitialiseCUDA() {
    char* initMem;
    cudaMalloc(&initMem, sizeof(char));
    cudaFree(initMem);
}

bool SortByColour(colour px1[3], colour px2[3]) {
    if (px1[0] > px2[0]) return true;
    else if (px1[0] < px2[0]) return false;
    else if (px1[1] > px2[1]) return true;
    else if (px1[1] < px2[1]) return false;
    else return px1[2] > px2[2];
}


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