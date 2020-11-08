#include "defines.hpp"

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "colour_indexer.cuh"

using namespace std;

bool ColourEquals(colour* a, colour* b) {
    bool bEq = a[0] == b[0];
    bool gEq = a[1] == b[1];
    bool rEq = a[2] == b[2];
    return bEq && gEq && rEq;
}

__device__
bool ColourEqualsGPU(colour* a, colour* b) {
    bool bEq = a[0] == b[0];
    bool gEq = a[1] == b[1];
    bool rEq = a[2] == b[2];
    return bEq && gEq && rEq;
}

__global__
void IndexColours(colour* colours, int n, ColourEntry* indexer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int colIdx = i * 3;
        int idx = colours[colIdx] + (colours[colIdx + 1] * 256) + (colours[colIdx + 2] * 65536);
        //printf("i: %d - Colour: %d,%d,%d; Index: %d\n", i, colours[colIdx], colours[colIdx + 1], colours[colIdx + 2], idx);
        indexer[idx].occupied = full_index;
        //atomicExch((int*)&indexer[idx], full_index);
    }
}

__device__
bool IndexIntersects(int index, colour3* indexer) {
    uint32_t hashed_index = Murmur3HashIndex(index);
    while (true) {
        if (indexer[hashed_index] != empty_index) {
            // TODO
        }
        else {
            return false;
        }
    }
}

ColourEntry* initColourIndexer() {
    ColourEntry* indexer;
    cudaMalloc(&indexer, sizeof(ColourEntry) * indexer_capacity);
    cudaMemset(indexer, 0x00, sizeof(ColourEntry) * indexer_capacity);
    return indexer;
}

// Reverse pixel generation algorithm to figure out to what pixel each excluded colour belongs
// By creating a hashed list of colours the 
void prepareExclusionList(ColourEntry* indexer, colour* exclusions, int size) {

    double t = (double)cv::getTickCount();

    colour* exclGPU;
    cudaMalloc(&exclGPU, size * sizeof(colour) * 3);
    cudaMemcpy(exclGPU, exclusions, sizeof(colour) * 3 * size, cudaMemcpyHostToDevice);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Exclusions copied in " << t << "s" << endl;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    cout << "Threads: " << blockSize << "x" << numBlocks << " = " << (blockSize * numBlocks) << " threads" << endl;

    t = (double)cv::getTickCount();

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Indexer initalised in " << t << "s" << endl;

    t = (double)cv::getTickCount();

    IndexColours <<<numBlocks, blockSize>>> (exclGPU, size, indexer);
    cudaDeviceSynchronize();

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Indexed in " << t << "s" << endl;

    cudaFree(exclGPU);

}