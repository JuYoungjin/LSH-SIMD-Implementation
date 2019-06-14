#include <stdio.h>
#include <stdlib.h>

#include "lsh_benchmark.h"
#include "lsh256_ref_AVX2_OPT.h"
#include "lsh512_ref_AVX2_OPT.h"
#include "benchmark.h"

unsigned int lsh256_benchmark(unsigned int databitlen)
{
	unsigned int calibration, tMin = 0xFFFFFFFF, t0, t1;
	int i;

	unsigned char hashval256[32];
	unsigned char data[TEST_MESSAGE_SIZE];
	
	if ((databitlen & 0x7) || (databitlen > (TEST_MESSAGE_SIZE << 3))){
		printf("databitlen must be a multiple of 8 and less than %d\n", TEST_MESSAGE_SIZE << 3);
		return 0;
	}	

	calibration = calibrate();

	for (i=0; i<TIMER_SAMPLE_CNT;i++)  
	{
		t0 = HiResTime();

		/*	function for test	*/				
		Hash256(256,data, databitlen, hashval256);
		/*	function for test	*/

		t1 = HiResTime();

		if (tMin > t1 - t0 - calibration)       /* keep only the minimum time */
				tMin = t1 - t0 - calibration;
	}
	
	return tMin;
}
unsigned int lsh512_benchmark(unsigned int databitlen)
{
	unsigned int calibration, tMin = 0xFFFFFFFF, t0, t1;
	int i;

	unsigned char hashval512[64];
	unsigned char data[TEST_MESSAGE_SIZE];

	if ((databitlen & 0x7) || (databitlen > (TEST_MESSAGE_SIZE << 3))){
		printf("databitlen must be a multiple of 8 and less than %d\n", TEST_MESSAGE_SIZE << 3);
		return 0;
	}

	calibration = calibrate();	


	for (i = 0; i<TIMER_SAMPLE_CNT; i++)  
	{
		t0 = HiResTime();

		/*	function for test	*/
		Hash512(512, data, databitlen, hashval512);
		/*	function for test	*/

		t1 = HiResTime();

		if (tMin > t1 - t0 - calibration)       /* keep only the minimum time */
			tMin = t1 - t0 - calibration;
	}
	
	return tMin;
}




