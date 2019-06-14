#include <stdlib.h>
#include <stdio.h>
#include "lsh256_ref.c"
#include "lsh512_ref.c"
#include "benchmark.c"
#include "lsh_benchmark.c"

void lsh_benchmark(){

	unsigned int tMin;

	printf("\nLSH Benchmark Results REF:\n\n");

	// for 64-byte messages
	tMin = lsh256_benchmark(0x200);
	printf("LSH-256-256,   64-byte messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 64));

	// for 1024-byte messages
	tMin = lsh256_benchmark(0x2000);
	printf("LSH-256-256, 1024-byte messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 1024));

	// for 4096-byte messages
	tMin = lsh256_benchmark(0x8000);
	printf("LSH-256-256, 4096-byte messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 4096));

	//for long messages
	tMin = (lsh256_benchmark(0x8000) - lsh256_benchmark(0x4000));
	printf("LSH-256-256,      long messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 2048));
 
	// for 64-byte messages
	tMin = lsh512_benchmark(0x200);
	printf("LSH-512-512,   64-byte messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 64));

	// for 1024-byte messages
	tMin = lsh512_benchmark(0x2000);
	printf("LSH-512-512, 1024-byte messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 1024));

	// for 4096-byte messages
	tMin = lsh512_benchmark(0x8000);
	printf("LSH-512-512, 4096-byte messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 4096));

	//for long messages
	tMin = (lsh512_benchmark(0x8000) - lsh512_benchmark(0x4000));
	printf("LSH-512-512,      long messages: ");
	printf("%7.2f cycles/byte\n", get_cpb(tMin, 2048));	

	return;
}

int main(int argc, char *argv[]){
	u8 data[1024/8] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
			0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,
			0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2a,0x2b,0x2c,0x2d,0x2e,0x2f,
			0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x3b,0x3c,0x3d,0x3e,0x3f,
			0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x4b,0x4c,0x4d,0x4e,0x4f,
			0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x5b,0x5c,0x5d,0x5e,0x5f,
			0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x6b,0x6c,0x6d,0x6e,0x6f,
			0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x7b,0x7c,0x7d,0x7e,0x7f};
	int i;
	printf("Hash Test\n");
	Hash256(224,(unsigned char * )data, 448, (unsigned char * )data);
	
	for(i=0;i<224/8;i++)
	{
		printf("%02x",data[i]);
	}
	printf("\n");

	Hash256(256,(unsigned char * )data, 512, (unsigned char * )data);
	
	for(i=0;i<256/8;i++)
	{
		printf("%02x",data[i]);
	}
	printf("\n");

	Hash512(384,(unsigned char * )data, 768, (unsigned char * )data);
	
	for(i=0;i<384/8;i++)
	{
		printf("%02x",data[i]);
	}

	printf("\n");

	Hash512(512,(unsigned char * )data, 1024, (unsigned char * )data);
	
	for(i=0;i<512/8;i++)
	{
		printf("%02x",data[i]);
	}
	printf("\n");

	lsh_benchmark();
		//getchar();
	return 0;
}