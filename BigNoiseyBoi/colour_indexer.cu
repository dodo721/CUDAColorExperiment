#include "defines.hpp"

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "imgutils.hpp"
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

void IndexColours (colour* colours, int n, ColourEntry* indexer) {
    int indexCount = 0;
    for (int i = 0; i < n; i ++) {
        int colIdx = i * 3;
        int idx = colours[colIdx] + (colours[colIdx + 1] * 256) + (colours[colIdx + 2] * 65536);
        //printf("i: %d - Colour: %d,%d,%d; Index: %d\n", i, colours[colIdx], colours[colIdx + 1], colours[colIdx + 2], idx);
        indexer[idx].occupied = full_index;
        indexer[idx].order = indexCount;
        indexCount++;
        //atomicExch((int*)&indexer[idx], full_index);
    }
}

ColourEntry* initColourIndexer(ColourEntry* indexer_cpu) {
    ColourEntry* indexer;
    int indexer_size = sizeof(ColourEntry) * indexer_capacity;
    cudaMalloc(&indexer, indexer_size);
    cudaMemcpy(indexer, indexer_cpu, indexer_size, cudaMemcpyHostToDevice);
    return indexer;
}

// Reverse pixel generation algorithm to figure out to what pixel each excluded colour belongs
ColourEntry* prepareExclusionList(colour* exclusions, int size, bool gpu) {

    // Create shared memory indexer for indexing operations
    ColourEntry* indexer_cpu = new ColourEntry[indexer_capacity];

    IndexColours (exclusions, size, indexer_cpu);

    if (!gpu) return indexer_cpu;

    // Create GPU-only indexer for use in image gen
    ColourEntry* indexer = initColourIndexer(indexer_cpu);

    delete[] indexer_cpu;

    return indexer;

}

ColourEntry* CreateIndex(colour* exclusions, size_t exclLength, bool gpu, bool verbose) {
    if (verbose) cout << "Indexing " << exclLength << " colours" << endl;
    ColourEntry* exclusionIndex = nullptr;
    if (exclusions != nullptr) {
        double t = cv::getTickCount();
        exclusionIndex = prepareExclusionList(exclusions, exclLength, gpu);
        t = (cv::getTickCount() - t) / cv::getTickFrequency();
        if (verbose) cout << "Exclusions indexed in " << t << endl;
    }
    return exclusionIndex;
}

void FreeIndex(ColourEntry* index, bool gpu) {
    if (gpu)
        cudaFree(index);
    else
        delete[] index;
}