#include "defines.hpp"

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "hashed_indexer.cuh"

using namespace std;

// 64 bit FNV1a hash
__device__
uint64_t FNV1a64HashIndex(int idx)
{
    const uint64_t prime = 1099511628211;
    uint64_t hash = 14695981039346656037;
    for (int j = 0; j < sizeof(int); j++) {
        char byte = *(&idx + j);
        hash ^= byte;
        hash *= prime;
    }
    return hash;
}

// 32 bit Murmur3 hash
__device__
uint32_t Murmur3HashIndex(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (indexer_capacity - 1);
}

bool ColourEquals(colour* a, colour* b) {
    int diff = (a[0] - b[0]) + (a[1] - b[1]) + (a[2] - b[2]);
    return diff == 0;
}

__device__
bool ColourEqualsGPU(colour* a, colour* b) {
    int diff = (a[0] - b[0]) + (a[1] - b[1]) + (a[2] - b[2]);
    return diff == 0;
}

__global__
void IndexColours(colour* colours, int n, bool* indexer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int colIdx = i * 3;
        int idx = colours[colIdx] + (colours[colIdx + 1] * 256) + (colours[colIdx + 2] * 65536);
        //printf("i: %d - Colour: %d,%d,%d; Index: %d\n", i, colours[colIdx], colours[colIdx + 1], colours[colIdx + 2], idx);
        indexer[idx] = full_index;
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

bool* initColourIndexer() {
    bool* indexer;
    cudaMalloc(&indexer, sizeof(colour3) * indexer_capacity);
    cudaMemset(indexer, empty_index, sizeof(bool) * indexer_capacity);
    return indexer;
}

// Reverse pixel generation algorithm to figure out to what pixel each excluded colour belongs
// By creating a hashed list of colours the 
void prepareExclusionList(bool* indexer, colour* exclusions, int size) {

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