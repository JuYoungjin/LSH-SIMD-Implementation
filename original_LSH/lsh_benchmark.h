#ifndef _LSH_BENCHMARK_H_
#define _LSH_BENCHMARK_H_
#include <stdint.h>

#define TEST_MESSAGE_SIZE			320000

unsigned int lsh256_benchmark(unsigned int databitlen);
unsigned int lsh512_benchmark(unsigned int databitlen);

#endif