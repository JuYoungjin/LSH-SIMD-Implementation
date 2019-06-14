#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "lsh256_ref_SSE_OPT.h"
#include <tmmintrin.h>

#define TwoSTEPs256(j) \
	TT[0] = _mm_xor_si128( TT[0], M[0]);\
	TT[1] = _mm_xor_si128( TT[1], M[1]);\
	TT[2] = _mm_xor_si128( TT[2], M[2]);\
	TT[3] = _mm_xor_si128( TT[3], M[3]);\
	\
	TT[0] = _mm_add_epi32( TT[0], TT[2]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 29), _mm_srli_epi32( TT[0], 3));\
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC256[j][0])));\
	\
	TT[1] = _mm_add_epi32( TT[1], TT[3]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 29), _mm_srli_epi32( TT[1], 3));\
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[j][4])));\
	\
	TT[2] = _mm_add_epi32( TT[2], TT[0]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 1), _mm_srli_epi32( TT[2], 31));\
	\
	TT[3] = _mm_add_epi32( TT[3], TT[1]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 1), _mm_srli_epi32( TT[3], 31));\
	\
	TT[0] = _mm_add_epi32( TT[0], TT[2]);\
	TT[0] = _mm_shuffle_epi32( TT[0], 210);\
	TT[1] = _mm_add_epi32( TT[1], TT[3]);\
	\
	TT[2] = _mm_shuffle_epi8(TT[2],G1);\
	TT[3] = _mm_shuffle_epi8(TT[3],G2);\
	\
	\
	M[0] = _mm_shuffle_epi32( M[0], 75);\
	M[0] = _mm_add_epi32(M[0], M[4]);\
	\
	M[1] = _mm_shuffle_epi32( M[1], 30);\
	M[1] = _mm_add_epi32(M[1], M[5]);\
	\
	M[2] = _mm_shuffle_epi32( M[2], 75);\
	M[2] = _mm_add_epi32(M[2], M[6]);\
	\
	M[3] = _mm_shuffle_epi32( M[3], 30);\
	M[3] = _mm_add_epi32(M[3], M[7]);\
	\
	\
	TT[1] = _mm_xor_si128( TT[1], M[4]);\
	TT[3] = _mm_xor_si128( TT[3], M[5]);\
	TT[0] = _mm_xor_si128( TT[0], M[6]);\
	TT[2] = _mm_xor_si128( TT[2], M[7]);\
	\
	TT[1] = _mm_add_epi32( TT[1], TT[0]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 5), _mm_srli_epi32( TT[1], 27));\
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[j+1][0])));\
	\
	TT[3] = _mm_add_epi32( TT[3], TT[2]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 5), _mm_srli_epi32( TT[3], 27));\
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC256[j+1][4])));\
	\
	TT[0] = _mm_add_epi32( TT[0], TT[1]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 17), _mm_srli_epi32( TT[0], 15));\
	\
	TT[2] = _mm_add_epi32( TT[2], TT[3]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 17), _mm_srli_epi32( TT[2], 15));\
	\
	TT[1] = _mm_add_epi32( TT[1], TT[0]);\
	TT[3] = _mm_add_epi32( TT[3], TT[2]);\
	Temp  = _mm_shuffle_epi32( TT[1], 210);\
	TT[1] = _mm_shuffle_epi8(TT[2],G2);\
	TT[2] = Temp;\
	Temp = TT[3];\
	TT[3] = _mm_shuffle_epi8(TT[0],G1);\
	TT[0] = Temp;\
	\
	\
	M[4] = _mm_shuffle_epi32( M[4], 75);\
	M[4] = _mm_add_epi32(M[4], M[0]);\
	\
	M[5] = _mm_shuffle_epi32( M[5], 30);\
	M[5] = _mm_add_epi32(M[5], M[1]);\
	\
	M[6] = _mm_shuffle_epi32( M[6], 75);\
	M[6] = _mm_add_epi32(M[6], M[2]);\
	\
	M[7] = _mm_shuffle_epi32( M[7], 30);\
	M[7] = _mm_add_epi32(M[7], M[3]);\
	

#define TwoSTEPs256_wo_ME(j) \
	TT[0] = _mm_add_epi32( TT[0], TT[2]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 29), _mm_srli_epi32( TT[0], 3));\
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC256[j][0])));\
	\
	TT[1] = _mm_add_epi32( TT[1], TT[3]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 29), _mm_srli_epi32( TT[1], 3));\
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[j][4])));\
	\
	TT[2] = _mm_add_epi32( TT[2], TT[0]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 1), _mm_srli_epi32( TT[2], 31));\
	\
	TT[3] = _mm_add_epi32( TT[3], TT[1]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 1), _mm_srli_epi32( TT[3], 31));\
	\
	TT[0] = _mm_add_epi32( TT[0], TT[2]);\
	TT[0] = _mm_shuffle_epi32( TT[0], 210);\
	TT[1] = _mm_add_epi32( TT[1], TT[3]);\
	\
	TT[2] = _mm_shuffle_epi8(TT[2],G1);\
	TT[3] = _mm_shuffle_epi8(TT[3],G2);\
	\
	\
	TT[1] = _mm_add_epi32( TT[1], TT[0]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 5), _mm_srli_epi32( TT[1], 27));\
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[j+1][0])));\
	\
	TT[3] = _mm_add_epi32( TT[3], TT[2]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 5), _mm_srli_epi32( TT[3], 27));\
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC256[j+1][4])));\
	\
	TT[0] = _mm_add_epi32( TT[0], TT[1]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 17), _mm_srli_epi32( TT[0], 15));\
	\
	TT[2] = _mm_add_epi32( TT[2], TT[3]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 17), _mm_srli_epi32( TT[2], 15));\
	\
	TT[1] = _mm_add_epi32( TT[1], TT[0]);\
	TT[3] = _mm_add_epi32( TT[3], TT[2]);\
	Temp  = _mm_shuffle_epi32( TT[1], 210);\
	TT[1] = _mm_shuffle_epi8(TT[2],G2);\
	TT[2] = Temp;\
	Temp = TT[3];\
	TT[3] = _mm_shuffle_epi8(TT[0],G1);\
	TT[0] = Temp;\


void compress256(hashState256 * state, const Byte * datablock) {
	ALIGN(16) u32 m[32];
	ALIGN(16) const u32 Gamma1[4] = {0x09080b0a,0x03020100,0x0c0f0e0d,0x06050407};
	ALIGN(16) const u32 Gamma2[4] = {0x02010003,0x04070605,0x0f0e0d0c,0x09080b0a};
	__m128i G1 = _mm_load_si128((__m128i*)Gamma1);
	__m128i G2 = _mm_load_si128((__m128i*)Gamma2);
	__m128i Temp;
	__m128i M[8];
	__m128i TT[4];
	
	m[ 0] = U8TO32_LE(datablock);		m[16] = U8TO32_LE(datablock + 64);    
	m[ 1] = U8TO32_LE(datablock +  4);	m[17] = U8TO32_LE(datablock +  4+ 64);
	m[ 2] = U8TO32_LE(datablock +  8);	m[18] = U8TO32_LE(datablock +  8+ 64);
	m[ 3] = U8TO32_LE(datablock + 12);	m[19] = U8TO32_LE(datablock + 12+ 64);
	m[ 4] = U8TO32_LE(datablock + 24);	m[20] = U8TO32_LE(datablock + 24+ 64);
	m[ 5] = U8TO32_LE(datablock + 16);	m[21] = U8TO32_LE(datablock + 16+ 64);
	m[ 6] = U8TO32_LE(datablock + 20);	m[22] = U8TO32_LE(datablock + 20+ 64);
	m[ 7] = U8TO32_LE(datablock + 28);	m[23] = U8TO32_LE(datablock + 28+ 64);
	m[ 8] = U8TO32_LE(datablock + 32);	m[24] = U8TO32_LE(datablock + 32+ 64);
	m[ 9] = U8TO32_LE(datablock + 36);	m[25] = U8TO32_LE(datablock + 36+ 64);
	m[10] = U8TO32_LE(datablock + 40);	m[26] = U8TO32_LE(datablock + 40+ 64);
	m[11] = U8TO32_LE(datablock + 44);	m[27] = U8TO32_LE(datablock + 44+ 64);
	m[12] = U8TO32_LE(datablock + 56);	m[28] = U8TO32_LE(datablock + 56+ 64);
	m[13] = U8TO32_LE(datablock + 48);	m[29] = U8TO32_LE(datablock + 48+ 64);
	m[14] = U8TO32_LE(datablock + 52);	m[30] = U8TO32_LE(datablock + 52+ 64);
	m[15] = U8TO32_LE(datablock + 60);	m[31] = U8TO32_LE(datablock + 60+ 64);
	
	M[0] = _mm_load_si128((__m128i*)m);
	M[1] = _mm_load_si128((__m128i*)(m+4));
	M[2] = _mm_load_si128((__m128i*)(m+8));
	M[3] = _mm_load_si128((__m128i*)(m+12));
	M[4] = _mm_load_si128((__m128i*)(m+16));
	M[5] = _mm_load_si128((__m128i*)(m+20));
	M[6] = _mm_load_si128((__m128i*)(m+24));
	M[7] = _mm_load_si128((__m128i*)(m+28));

	TT[0] = _mm_load_si128((__m128i*)state->cv256);
	TT[1] = _mm_load_si128((__m128i*)(state->cv256+4));
	TT[2] = _mm_load_si128((__m128i*)(state->cv256+8));
	TT[3] = _mm_load_si128((__m128i*)(state->cv256+12));
	
	TwoSTEPs256(0);
	TwoSTEPs256(2);
	TwoSTEPs256(4);
	TwoSTEPs256(6);
	TwoSTEPs256(8);
	TwoSTEPs256(10);
	TwoSTEPs256(12);
	TwoSTEPs256(14);
	TwoSTEPs256(16);
	TwoSTEPs256(18);
	TwoSTEPs256(20);
	TwoSTEPs256(22);
	
	//24 STEP start
	TT[0] = _mm_xor_si128( TT[0], M[0]);
	TT[1] = _mm_xor_si128( TT[1], M[1]);
	TT[2] = _mm_xor_si128( TT[2], M[2]);
	TT[3] = _mm_xor_si128( TT[3], M[3]);
	
	TT[0] = _mm_add_epi32( TT[0], TT[2]);
	TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 29), _mm_srli_epi32( TT[0], 3));
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC256[24][0])));
	
	TT[1] = _mm_add_epi32( TT[1], TT[3]);
	TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 29), _mm_srli_epi32( TT[1], 3));
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[24][4])));
	
	TT[2] = _mm_add_epi32( TT[2], TT[0]);
	TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 1), _mm_srli_epi32( TT[2], 31));
	
	TT[3] = _mm_add_epi32( TT[3], TT[1]);
	TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 1), _mm_srli_epi32( TT[3], 31));
	
	TT[0] = _mm_add_epi32( TT[0], TT[2]);
	TT[0] = _mm_shuffle_epi32( TT[0], 210);
	TT[1] = _mm_add_epi32( TT[1], TT[3]);

	TT[2] = _mm_shuffle_epi8(TT[2],G1);
	TT[3] = _mm_shuffle_epi8(TT[3],G2);
	
	M[0] = _mm_shuffle_epi32( M[0], 75);
	M[0] = _mm_add_epi32(M[0], M[4]);
	
	M[1] = _mm_shuffle_epi32( M[1], 30);
	M[1] = _mm_add_epi32(M[1], M[5]);
	
	M[2] = _mm_shuffle_epi32( M[2], 75);
	M[2] = _mm_add_epi32(M[2], M[6]);
	
	M[3] = _mm_shuffle_epi32( M[3], 30);
	M[3] = _mm_add_epi32(M[3], M[7]);

	//25 STEP start	
	TT[1] = _mm_xor_si128( TT[1], M[4]);
	TT[3] = _mm_xor_si128( TT[3], M[5]);
	TT[0] = _mm_xor_si128( TT[0], M[6]);
	TT[2] = _mm_xor_si128( TT[2], M[7]);
	
	TT[1] = _mm_add_epi32( TT[1], TT[0]);
	TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 5), _mm_srli_epi32( TT[1], 27));
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[25][0])));
	
	TT[3] = _mm_add_epi32( TT[3], TT[2]);
	TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 5), _mm_srli_epi32( TT[3], 27));
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC256[25][4])));
	
	TT[0] = _mm_add_epi32( TT[0], TT[1]);
	TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 17), _mm_srli_epi32( TT[0], 15));
	
	TT[2] = _mm_add_epi32( TT[2], TT[3]);
	TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 17), _mm_srli_epi32( TT[2], 15));
	
	TT[1] = _mm_add_epi32( TT[1], TT[0]);
	TT[1] = _mm_shuffle_epi32( TT[1], 210);
	TT[1] = _mm_xor_si128( TT[1], M[2]);
	TT[3] = _mm_add_epi32( TT[3], TT[2]);
	TT[3] = _mm_xor_si128( TT[3], M[0]);
	
	TT[0] = _mm_shuffle_epi8(TT[0],G1);
	TT[0] = _mm_xor_si128( TT[0], M[3]);
	
	TT[2] = _mm_shuffle_epi8(TT[2],G2);
	TT[2] = _mm_xor_si128( TT[2], M[1]);
	
	//store
	_mm_store_si128((__m128i*)(&state->cv256[0]),TT[3]);
	_mm_store_si128((__m128i*)(&state->cv256[4]),TT[2]);
	_mm_store_si128((__m128i*)(&state->cv256[8]),TT[1]);
	_mm_store_si128((__m128i*)(&state->cv256[12]),TT[0]);


	return;
}
int Init256(hashState256 * state, int hashbitlen) {
	int j;
	
	if (hashbitlen == 256)
	{
		memcpy(state->cv256, IV256, sizeof(IV256));
	}
	else if ((hashbitlen <0) || (hashbitlen>256))
	{
		return FAIL;
	}
	else
	{
		memset(state->cv256, 0, 16 * sizeof(u32));
		state->cv256[0] = 32;
		state->cv256[1] = (u32)hashbitlen;

		ALIGN(16) const u32 Gamma1[4] = {0x09080b0a,0x03020100,0x0c0f0e0d,0x06050407};
		ALIGN(16) const u32 Gamma2[4] = {0x02010003,0x04070605,0x0f0e0d0c,0x09080b0a};
		__m128i G1 = _mm_load_si128((__m128i*)Gamma1);
		__m128i G2 = _mm_load_si128((__m128i*)Gamma2);
		__m128i Temp;
		__m128i TT[4];
		
		TT[0] = _mm_load_si128((__m128i*)state->cv256);
		TT[1] = _mm_load_si128((__m128i*)(state->cv256+4));
		TT[2] = _mm_load_si128((__m128i*)(state->cv256+8));
		TT[3] = _mm_load_si128((__m128i*)(state->cv256+12));
		
		TwoSTEPs256_wo_ME(0);
		TwoSTEPs256_wo_ME(2);
		TwoSTEPs256_wo_ME(4);
		TwoSTEPs256_wo_ME(6);
		TwoSTEPs256_wo_ME(8);
		TwoSTEPs256_wo_ME(10);
		TwoSTEPs256_wo_ME(12);
		TwoSTEPs256_wo_ME(14);
		TwoSTEPs256_wo_ME(16);
		TwoSTEPs256_wo_ME(18);
		TwoSTEPs256_wo_ME(20);
		TwoSTEPs256_wo_ME(22);
		
		//24 STEP start
		TT[0] = _mm_add_epi32( TT[0], TT[2]);
		TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 29), _mm_srli_epi32( TT[0], 3));
		TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC256[24][0])));
		
		TT[1] = _mm_add_epi32( TT[1], TT[3]);
		TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 29), _mm_srli_epi32( TT[1], 3));
		TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[24][4])));
		
		TT[2] = _mm_add_epi32( TT[2], TT[0]);
		TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 1), _mm_srli_epi32( TT[2], 31));
		
		TT[3] = _mm_add_epi32( TT[3], TT[1]);
		TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 1), _mm_srli_epi32( TT[3], 31));
		
		TT[0] = _mm_add_epi32( TT[0], TT[2]);
		TT[0] = _mm_shuffle_epi32( TT[0], 210);
		TT[1] = _mm_add_epi32( TT[1], TT[3]);

		TT[2] = _mm_shuffle_epi8(TT[2],G1);
		TT[3] = _mm_shuffle_epi8(TT[3],G2);

		//25 STEP start	
		TT[1] = _mm_add_epi32( TT[1], TT[0]);
		TT[1] = _mm_xor_si128(_mm_slli_epi32( TT[1], 5), _mm_srli_epi32( TT[1], 27));
		TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC256[25][0])));
		
		TT[3] = _mm_add_epi32( TT[3], TT[2]);
		TT[3] = _mm_xor_si128(_mm_slli_epi32( TT[3], 5), _mm_srli_epi32( TT[3], 27));
		TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC256[25][4])));
		
		TT[0] = _mm_add_epi32( TT[0], TT[1]);
		TT[0] = _mm_xor_si128(_mm_slli_epi32( TT[0], 17), _mm_srli_epi32( TT[0], 15));
		
		TT[2] = _mm_add_epi32( TT[2], TT[3]);
		TT[2] = _mm_xor_si128(_mm_slli_epi32( TT[2], 17), _mm_srli_epi32( TT[2], 15));
		
		TT[1] = _mm_add_epi32( TT[1], TT[0]);
		TT[1] = _mm_shuffle_epi32( TT[1], 210);
		TT[3] = _mm_add_epi32( TT[3], TT[2]);
		
		TT[0] = _mm_shuffle_epi8(TT[0],G1);
		
		TT[2] = _mm_shuffle_epi8(TT[2],G2);
		
		//store
		_mm_store_si128((__m128i*)(&state->cv256[0]),TT[3]);
		_mm_store_si128((__m128i*)(&state->cv256[4]),TT[2]);
		_mm_store_si128((__m128i*)(&state->cv256[8]),TT[1]);
		_mm_store_si128((__m128i*)(&state->cv256[12]),TT[0]);
	}
	
	state->hashbitlen = hashbitlen;
	return SUCCESS;
}
void Update256(hashState256 * state, const Byte * data, DataLength databitlen) {

	u64 numBlocks, temp;
	u32 pos1, pos2;
	int i;
	numBlocks = ((u64)databitlen >> 10);	

	for (i = 0; i < numBlocks; i++){
		compress256(state, data);
		data += 128;
	}

	//computation of the state->Last256 block (padded)
	//if databitlen not multiple of 1024
	/*
	databitlen = 1024*(numBlocks) + 8*pos1 + pos2,
	0<=pos1<128, 0<=pos2<7
	*/
	if ((u32)(databitlen & 0x3ff)){
		temp = (numBlocks) << 7; //temp = 128*(numBlocks)
		pos1 = (u32)((databitlen >> 3) - temp);
		pos2 = (u32)(databitlen & 0x7);

		//if databitlen not multiple of 8
		if (pos2){
			memcpy(state->Last256, data, pos1*sizeof(char));
			state->Last256[pos1] = (data[pos1] & (0xff << (8 - pos2))) ^ (1 << (7 - pos2));
			if (pos1 != 127) memset(state->Last256 + pos1 + 1, 0, (127 - pos1)*sizeof(char));
		}
		//if databitlen multiple of 8
		else{
			memcpy(state->Last256, data, pos1*sizeof(char));
			state->Last256[pos1] = 0x80;
			if (pos1 != 127) memset(state->Last256 + pos1 + 1, 0, (127 - pos1)*sizeof(char));
		}
	}
	//if databitlen multiple of 1024
	else{
		state->Last256[0] = 0x80;
		memset(state->Last256 + 1, 0, 127 * sizeof(u8));
	}
	// end: computation of the state->Last256 block

	return;
}
void Final256(hashState256 * state, Byte * hashval) {
	int l;
	u32 H[8];

	compress256(state, state->Last256);

	H[0] = (state->cv256[0]) ^ (state->cv256[8]);
	H[1] = (state->cv256[1]) ^ (state->cv256[9]);
	H[2] = (state->cv256[2]) ^ (state->cv256[10]);
	H[3] = (state->cv256[3]) ^ (state->cv256[11]);
	H[4] = (state->cv256[5]) ^ (state->cv256[13]);
	H[5] = (state->cv256[6]) ^ (state->cv256[14]);
	H[6] = (state->cv256[4]) ^ (state->cv256[12]);
	H[7] = (state->cv256[7]) ^ (state->cv256[15]);

	for (l = 0; l < (state->hashbitlen) >> 3; l++){
//		hashval[l] = (u8)(ROR32(H[l >> 2], (l << 3) & 0x1f) ); 
		hashval[l] = (u8)(H[l >> 2] >> ((l << 3) & 0x1f)); //0,8,16,24,0,,.. = 8*l (mod 32) = (l<<3)&0x1f
	}

	return;
}
int Hash256(int hashbitlen, const Byte * data, DataLength databitlen, Byte * hashval) {

	int ret;
	hashState256 state;

	if (data == NULL && databitlen > 0){
		return FAIL;
	}
	if (hashval == NULL){
		return FAIL;
	}

	ret = Init256(&state, hashbitlen);
	if (ret != SUCCESS) return ret;

	Update256(&state, data, databitlen);	

	Final256(&state, hashval);

	return SUCCESS;
}

