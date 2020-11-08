#ifndef COLOUR_INDEXER_CUH
#define COLOUR_INDEXER_CUH

#include <stdint.h>

struct ColourEntry {
public:
	bool occupied = false;
	unsigned int order = 0;
};

const uint32_t indexer_capacity = 256 * 256 * 256;
const bool empty_index = false;
const bool full_index = true;
ColourEntry* prepareExclusionList(colour* exclusions, int size);
bool ColourEquals(colour* a, colour* b);
bool ColourEqualsGPU(colour* a, colour* b);

#endif