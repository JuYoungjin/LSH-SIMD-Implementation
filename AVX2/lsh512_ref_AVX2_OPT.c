#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "lsh512_ref_AVX2_OPT.h"
#include <immintrin.h>

#define TwoSTEPs512(j) \
	TT[0] = _mm256_xor_si256( TT[0], M[0]);\
	TT[1] = _mm256_xor_si256( TT[1], M[1]);\
	TT[2] = _mm256_xor_si256( TT[2], M[2]);\
	TT[3] = _mm256_xor_si256( TT[3], M[3]);\
	\
	TT[0] = _mm256_add_epi64( TT[0], TT[2]);\
	TT[1] = _mm256_add_epi64( TT[1], TT[3]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 23), _mm256_srli_epi64( TT[0], 41));\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 23), _mm256_srli_epi64( TT[1], 41));\
	\
	TT[0] = _mm256_xor_si256( TT[0], _mm256_load_si256((__m256i*)(&SC512[j][0])));\
	TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[j][4])));\
	\
	TT[2] = _mm256_add_epi64( TT[2], TT[0]);\
	TT[3] = _mm256_add_epi64( TT[3], TT[1]);\
	\
	TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 59), _mm256_srli_epi64( TT[2], 5));\
	TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 59), _mm256_srli_epi64( TT[3], 5));\
	\
	TT[0] = _mm256_add_epi64( TT[0], TT[2]);\
	TT[0] = _mm256_permute4x64_epi64( TT[0], 210);\
	TT[1] = _mm256_add_epi64( TT[1], TT[3]);\
	\
	TT[2] = _mm256_permute4x64_epi64( TT[2], 114);\
	TT[2] = _mm256_shuffle_epi8(TT[2],G1);\
	TT[3] = _mm256_shuffle_epi8(TT[3],G2);\
	\
	\
	TT[1] = _mm256_xor_si256( TT[1], M[4]);\
	TT[3] = _mm256_xor_si256( TT[3], M[5]);\
	TT[0] = _mm256_xor_si256( TT[0], M[6]);\
	TT[2] = _mm256_xor_si256( TT[2], M[7]);\
	\
	TT[1] = _mm256_add_epi64( TT[1], TT[0]);\
	TT[3] = _mm256_add_epi64( TT[3], TT[2]);\
	\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 7), _mm256_srli_epi64( TT[1], 57));\
	TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 7), _mm256_srli_epi64( TT[3], 57));\
	\
	TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[j+1][0])));\
	TT[3] = _mm256_xor_si256( TT[3], _mm256_load_si256((__m256i*)(&SC512[j+1][4])));\
	\
	TT[0] = _mm256_add_epi64( TT[0], TT[1]);\
	TT[2] = _mm256_add_epi64( TT[2], TT[3]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 3), _mm256_srli_epi64( TT[0], 61));\
	TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 3), _mm256_srli_epi64( TT[2], 61));\
	\
	Temp2 = _mm256_add_epi64( TT[3], TT[2]);\
	TT[1] = _mm256_add_epi64( TT[1], TT[0]);\
	Temp = _mm256_permute4x64_epi64( TT[1], 210);\
	TT[1] = _mm256_shuffle_epi8(TT[2],G2);\
	TT[2] = Temp;\
	TT[0] = _mm256_permute4x64_epi64( TT[0], 114);\
	TT[3] = _mm256_shuffle_epi8(TT[0],G1);\
	TT[0] = Temp2;\
	\
	M[0] = _mm256_permute4x64_epi64( M[0], 75);\
	M[0] = _mm256_add_epi64(M[0], M[4]);\
	M[1] = _mm256_permute4x64_epi64( M[1], 30);\
	M[1] = _mm256_add_epi64(M[1], M[5]);\
	M[2] = _mm256_permute4x64_epi64( M[2], 75);\
	M[2] = _mm256_add_epi64(M[2], M[6]);\
	M[3] = _mm256_permute4x64_epi64( M[3], 30);\
	M[3] = _mm256_add_epi64(M[3], M[7]);\
	M[4] = _mm256_permute4x64_epi64( M[4], 75);\
	M[4] = _mm256_add_epi64(M[4], M[0]);\
	M[5] = _mm256_permute4x64_epi64( M[5], 30);\
	M[5] = _mm256_add_epi64(M[5], M[1]);\
	M[6] = _mm256_permute4x64_epi64( M[6], 75);\
	M[6] = _mm256_add_epi64(M[6], M[2]);\
	M[7] = _mm256_permute4x64_epi64( M[7], 30);\
	M[7] = _mm256_add_epi64(M[7], M[3]);\


#define TwoSTEPs512_wo_ME(j) \
	TT[0] = _mm256_add_epi64( TT[0], TT[2]);\
	TT[1] = _mm256_add_epi64( TT[1], TT[3]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 23), _mm256_srli_epi64( TT[0], 41));\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 23), _mm256_srli_epi64( TT[1], 41));\
	\
	TT[0] = _mm256_xor_si256( TT[0], _mm256_load_si256((__m256i*)(&SC512[j][0])));\
	TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[j][4])));\
	\
	TT[2] = _mm256_add_epi64( TT[2], TT[0]);\
	TT[3] = _mm256_add_epi64( TT[3], TT[1]);\
	\
	TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 59), _mm256_srli_epi64( TT[2], 5));\
	TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 59), _mm256_srli_epi64( TT[3], 5));\
	\
	TT[0] = _mm256_add_epi64( TT[0], TT[2]);\
	TT[0] = _mm256_permute4x64_epi64( TT[0], 210);\
	TT[1] = _mm256_add_epi64( TT[1], TT[3]);\
	\
	TT[2] = _mm256_permute4x64_epi64( TT[2], 114);\
	TT[2] = _mm256_shuffle_epi8(TT[2],G1);\
	TT[3] = _mm256_shuffle_epi8(TT[3],G2);\
	\
	\
	TT[1] = _mm256_add_epi64( TT[1], TT[0]);\
	TT[3] = _mm256_add_epi64( TT[3], TT[2]);\
	\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 7), _mm256_srli_epi64( TT[1], 57));\
	TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 7), _mm256_srli_epi64( TT[3], 57));\
	\
	TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[j+1][0])));\
	TT[3] = _mm256_xor_si256( TT[3], _mm256_load_si256((__m256i*)(&SC512[j+1][4])));\
	\
	TT[0] = _mm256_add_epi64( TT[0], TT[1]);\
	TT[2] = _mm256_add_epi64( TT[2], TT[3]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 3), _mm256_srli_epi64( TT[0], 61));\
	TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 3), _mm256_srli_epi64( TT[2], 61));\
	\
	Temp2 = _mm256_add_epi64( TT[3], TT[2]);\
	TT[1] = _mm256_add_epi64( TT[1], TT[0]);\
	Temp = _mm256_permute4x64_epi64( TT[1], 210);\
	TT[1] = _mm256_shuffle_epi8(TT[2],G2);\
	TT[2] = Temp;\
	TT[0] = _mm256_permute4x64_epi64( TT[0], 114);\
	TT[3] = _mm256_shuffle_epi8(TT[0],G1);\
	TT[0] = Temp2;\

void compress512(hashState512 * state, const Byte * datablock) {
	int i,j;
	ALIGN(32) u64 m[32];
	ALIGN(32) const u64 Gamma1[4] = 
		{0x0302010007060504ULL,0x0f0e0d0c0b0a0908ULL,0x0100070605040302ULL,0x0d0c0b0a09080f0eULL};
	ALIGN(32) const u64 Gamma2[4] = 
		{0x0201000706050403ULL,0x0e0d0c0b0a09080fULL,0x080f0e0d0c0b0a09ULL,0x0403020100070605ULL};
	__m256i G1 = _mm256_load_si256((__m256i*)Gamma1);
	__m256i G2 = _mm256_load_si256((__m256i*)Gamma2);
	__m256i M[8];
	__m256i TT[4];
	__m256i Temp, Temp2;
	
	for(i=0;i<32;i+=8)
	{ 
		j = i<<3;
		m[i  ] = U8TO64_LE(datablock + j     );
		m[i+1] = U8TO64_LE(datablock + j +  8);
		m[i+2] = U8TO64_LE(datablock + j + 16);
		m[i+3] = U8TO64_LE(datablock + j + 24);
		m[i+4] = U8TO64_LE(datablock + j + 48);
		m[i+5] = U8TO64_LE(datablock + j + 32);
		m[i+6] = U8TO64_LE(datablock + j + 40);
		m[i+7] = U8TO64_LE(datablock + j + 56);

	}

	M[0] = _mm256_load_si256((__m256i*)m);
	M[1] = _mm256_load_si256((__m256i*)(m+4));
	M[2] = _mm256_load_si256((__m256i*)(m+8));
	M[3] = _mm256_load_si256((__m256i*)(m+12));
	M[4] = _mm256_load_si256((__m256i*)(m+16));
	M[5] = _mm256_load_si256((__m256i*)(m+20));
	M[6] = _mm256_load_si256((__m256i*)(m+24));
	M[7] = _mm256_load_si256((__m256i*)(m+28));

	TT[0] = _mm256_load_si256((__m256i*)state->cv512);
	TT[1] = _mm256_load_si256((__m256i*)(state->cv512+4));
	TT[2] = _mm256_load_si256((__m256i*)(state->cv512+8));
	TT[3] = _mm256_load_si256((__m256i*)(state->cv512+12));
	
	for(i=0;i<24;i+=4)
	{
		TwoSTEPs512(i);
		TwoSTEPs512(i+2);
	}
	TwoSTEPs512(24);

	//26 STEP start
	TT[0] = _mm256_xor_si256( TT[0], M[0]);
	TT[1] = _mm256_xor_si256( TT[1], M[1]);
	TT[2] = _mm256_xor_si256( TT[2], M[2]);
	TT[3] = _mm256_xor_si256( TT[3], M[3]);
	
	TT[0] = _mm256_add_epi64( TT[0], TT[2]);
	TT[1] = _mm256_add_epi64( TT[1], TT[3]);
	
	TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 23), _mm256_srli_epi64( TT[0], 41));
	TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 23), _mm256_srli_epi64( TT[1], 41));
	
	TT[0] = _mm256_xor_si256( TT[0], _mm256_load_si256((__m256i*)(&SC512[26][0])));
	TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[26][4])));
	
	TT[2] = _mm256_add_epi64( TT[2], TT[0]);
	TT[3] = _mm256_add_epi64( TT[3], TT[1]);
	
	TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 59), _mm256_srli_epi64( TT[2], 5));
	TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 59), _mm256_srli_epi64( TT[3], 5));
	
	TT[0] = _mm256_add_epi64( TT[0], TT[2]);
	TT[0] = _mm256_permute4x64_epi64( TT[0], 210);
	TT[1] = _mm256_add_epi64( TT[1], TT[3]);
	
	TT[2] = _mm256_permute4x64_epi64( TT[2], 114);
	TT[2] = _mm256_shuffle_epi8(TT[2],G1);
	TT[3] = _mm256_shuffle_epi8(TT[3],G2);
	
	M[0] = _mm256_permute4x64_epi64( M[0], 75);
	M[0] = _mm256_add_epi64(M[0], M[4]);
	M[1] = _mm256_permute4x64_epi64( M[1], 30);
	M[1] = _mm256_add_epi64(M[1], M[5]);
	M[2] = _mm256_permute4x64_epi64( M[2], 75);
	M[2] = _mm256_add_epi64(M[2], M[6]);
	M[3] = _mm256_permute4x64_epi64( M[3], 30);
	M[3] = _mm256_add_epi64(M[3], M[7]);
	
	//27 STEP start	
	TT[1] = _mm256_xor_si256( TT[1], M[4]);
	TT[3] = _mm256_xor_si256( TT[3], M[5]);
	TT[0] = _mm256_xor_si256( TT[0], M[6]);
	TT[2] = _mm256_xor_si256( TT[2], M[7]);
	
	TT[1] = _mm256_add_epi64( TT[1], TT[0]);
	TT[3] = _mm256_add_epi64( TT[3], TT[2]);
	
	TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 7), _mm256_srli_epi64( TT[1], 57));
	TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 7), _mm256_srli_epi64( TT[3], 57));
	
	TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[27][0])));
	TT[3] = _mm256_xor_si256( TT[3], _mm256_load_si256((__m256i*)(&SC512[27][4])));
	
	TT[0] = _mm256_add_epi64( TT[0], TT[1]);
	TT[2] = _mm256_add_epi64( TT[2], TT[3]);
	
	TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 3), _mm256_srli_epi64( TT[0], 61));
	TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 3), _mm256_srli_epi64( TT[2], 61));
	
	TT[1] = _mm256_add_epi64( TT[1], TT[0]);
	TT[1] = _mm256_permute4x64_epi64( TT[1], 210);
	TT[1] = _mm256_xor_si256( TT[1], M[2]);
	TT[3] = _mm256_add_epi64( TT[3], TT[2]);
	TT[3] = _mm256_xor_si256( TT[3], M[0]);
	
	TT[0] = _mm256_permute4x64_epi64( TT[0], 114);
	TT[0] = _mm256_shuffle_epi8(TT[0],G1);
	TT[0] = _mm256_xor_si256( TT[0], M[3]);
	TT[2] = _mm256_shuffle_epi8(TT[2],G2);
	TT[2] = _mm256_xor_si256( TT[2], M[1]);

	_mm256_storeu_si256((__m256i*)(&state->cv512[0]),TT[3]);
	_mm256_storeu_si256((__m256i*)(&state->cv512[4]),TT[2]);
	_mm256_storeu_si256((__m256i*)(&state->cv512[8]),TT[1]);
	_mm256_storeu_si256((__m256i*)(&state->cv512[12]),TT[0]);

	return;
}
int Init512(hashState512 * state, int hashbitlen) {

	int j, l;
	int k;
	u64 vl, vr;

	u64 T[16];
	if(hashbitlen == 512)/* if hashbitlen == 512 */
	{
		memcpy(state->cv512, IV512, sizeof(IV512));
	}
	if ((hashbitlen <0) || (hashbitlen>512))
		return FAIL;
	else if (hashbitlen < 512){
		memset(state->cv512, 0, 16 * sizeof(u64));
		state->cv512[0] = 64;
		state->cv512[1] = (u64)hashbitlen;
		
		ALIGN(32) const u64 Gamma1[4] = 
			{0x0302010007060504ULL,0x0f0e0d0c0b0a0908ULL,0x0100070605040302ULL,0x0d0c0b0a09080f0eULL};
		ALIGN(32) const u64 Gamma2[4] = 
			{0x0201000706050403ULL,0x0e0d0c0b0a09080fULL,0x080f0e0d0c0b0a09ULL,0x0403020100070605ULL};
		__m256i G1 = _mm256_load_si256((__m256i*)Gamma1);
		__m256i G2 = _mm256_load_si256((__m256i*)Gamma2);
		__m256i TT[4];
		__m256i Temp, Temp2;
	
		TT[0] = _mm256_load_si256((__m256i*)state->cv512);
		TT[1] = _mm256_load_si256((__m256i*)(state->cv512+4));
		TT[2] = _mm256_load_si256((__m256i*)(state->cv512+8));
		TT[3] = _mm256_load_si256((__m256i*)(state->cv512+12));
		
		TwoSTEPs512_wo_ME(0);
		TwoSTEPs512_wo_ME(2);
		TwoSTEPs512_wo_ME(4);
		TwoSTEPs512_wo_ME(6);
		TwoSTEPs512_wo_ME(8);
		TwoSTEPs512_wo_ME(10);
		TwoSTEPs512_wo_ME(12);
		TwoSTEPs512_wo_ME(14);
		TwoSTEPs512_wo_ME(16);
		TwoSTEPs512_wo_ME(18);
		TwoSTEPs512_wo_ME(20);
		TwoSTEPs512_wo_ME(22);
		TwoSTEPs512_wo_ME(24);

		//26 STEP start
		TT[0] = _mm256_add_epi64( TT[0], TT[2]);
		TT[1] = _mm256_add_epi64( TT[1], TT[3]);
		
		TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 23), _mm256_srli_epi64( TT[0], 41));
		TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 23), _mm256_srli_epi64( TT[1], 41));
		
		TT[0] = _mm256_xor_si256( TT[0], _mm256_load_si256((__m256i*)(&SC512[26][0])));
		TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[26][4])));
		
		TT[2] = _mm256_add_epi64( TT[2], TT[0]);
		TT[3] = _mm256_add_epi64( TT[3], TT[1]);
		
		TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 59), _mm256_srli_epi64( TT[2], 5));
		TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 59), _mm256_srli_epi64( TT[3], 5));
		
		TT[0] = _mm256_add_epi64( TT[0], TT[2]);
		TT[0] = _mm256_permute4x64_epi64( TT[0], 210);
		TT[1] = _mm256_add_epi64( TT[1], TT[3]);
		
		TT[2] = _mm256_permute4x64_epi64( TT[2], 114);
		TT[2] = _mm256_shuffle_epi8(TT[2],G1);
		TT[3] = _mm256_shuffle_epi8(TT[3],G2);

		//27 STEP start	
		TT[1] = _mm256_add_epi64( TT[1], TT[0]);
		TT[3] = _mm256_add_epi64( TT[3], TT[2]);
		
		TT[1] = _mm256_xor_si256(_mm256_slli_epi64( TT[1], 7), _mm256_srli_epi64( TT[1], 57));
		TT[3] = _mm256_xor_si256(_mm256_slli_epi64( TT[3], 7), _mm256_srli_epi64( TT[3], 57));
		
		TT[1] = _mm256_xor_si256( TT[1], _mm256_load_si256((__m256i*)(&SC512[27][0])));
		TT[3] = _mm256_xor_si256( TT[3], _mm256_load_si256((__m256i*)(&SC512[27][4])));
		
		TT[0] = _mm256_add_epi64( TT[0], TT[1]);
		TT[2] = _mm256_add_epi64( TT[2], TT[3]);
		
		TT[0] = _mm256_xor_si256(_mm256_slli_epi64( TT[0], 3), _mm256_srli_epi64( TT[0], 61));
		TT[2] = _mm256_xor_si256(_mm256_slli_epi64( TT[2], 3), _mm256_srli_epi64( TT[2], 61));
		
		TT[1] = _mm256_add_epi64( TT[1], TT[0]);
		TT[1] = _mm256_permute4x64_epi64( TT[1], 210);
		TT[3] = _mm256_add_epi64( TT[3], TT[2]);
		
		TT[0] = _mm256_permute4x64_epi64( TT[0], 114);
		TT[0] = _mm256_shuffle_epi8(TT[0],G1);
		TT[2] = _mm256_shuffle_epi8(TT[2],G2);

		_mm256_storeu_si256((__m256i*)(&state->cv512[0]),TT[3]);
		_mm256_storeu_si256((__m256i*)(&state->cv512[4]),TT[2]);
		_mm256_storeu_si256((__m256i*)(&state->cv512[8]),TT[1]);
		_mm256_storeu_si256((__m256i*)(&state->cv512[12]),TT[0]);
	}

	state->hashbitlen = hashbitlen;
	return SUCCESS;
}
void Update512(hashState512 * state, const Byte * data, DataLength databitlen) {

	u64 numBlocks, temp;
	int pos1, pos2;
	int i;
	numBlocks = ((u64)databitlen >> 11);	

	for (i = 0; i < numBlocks; i++){
		compress512(state, data);
		data += 256;
	}
	
	//computation of the state->Last512 block (padded)
	//if databitlen not multiple of 2048
	//databitlen = 2048*(numBlocks) + 8*pos1 + pos2, 0<=pos1<256, 0<=pos2<7
	if ((u32)(databitlen)& 0x7ff){
		temp = (numBlocks) << 8; //temp = 256*(numBlocks)
		pos1 = (u32)((databitlen >> 3) - temp);
		pos2 = (u32)(databitlen & 0x7);

		//if databitlen not multiple of 8
		if (pos2){
			memcpy(state->Last512, data, pos1*sizeof(char));//			
			state->Last512[pos1] = (data[ pos1] & (0xff << (8 - pos2))) ^ (1 << (7 - pos2));
			if (pos1 != 255) memset(state->Last512 + pos1 + 1, 0, (255 - pos1)*sizeof(char));
		}
		//if databitlen multiple of 8
		else{
			memcpy(state->Last512, data, pos1*sizeof(char));
			state->Last512[pos1] = 0x80;
			if (pos1 != 255) memset(state->Last512 + pos1 + 1, 0, (255 - pos1)*sizeof(char));
		}
	}
	//if databitlen multiple of 2048
	else{
		state->Last512[0] = 0x80;
		memset(state->Last512 + 1, 0, 255 * sizeof(u8));
	}
	// end: computation of the state->Last512 block
	return;
}
void Final512(hashState512 * state, Byte * hashval) {
	int l;
	u64 H[8]; 

	compress512(state, state->Last512);

	H[0] = (state->cv512[0]) ^ (state->cv512[8]);
	H[1] = (state->cv512[1]) ^ (state->cv512[9]);
	H[2] = (state->cv512[2]) ^ (state->cv512[10]);
	H[3] = (state->cv512[3]) ^ (state->cv512[11]);
	H[4] = (state->cv512[5]) ^ (state->cv512[13]);
	H[5] = (state->cv512[6]) ^ (state->cv512[14]);
	H[6] = (state->cv512[4]) ^ (state->cv512[12]);
	H[7] = (state->cv512[7]) ^ (state->cv512[15]);

	for (l = 0; l < (state->hashbitlen) >> 3; l++){
//		hashval[l] = (u8)(ROR64(H[l >> 3], (l << 3) & 0x3f));
		hashval[l] = (u8)(H[l >> 3] >> ((l << 3) & 0x3f)); //0,8,16,24,32,40,48,56,0,... = 8*l (mod 64) = (l<<3)&0x3f
	}

	return;
}/////////////////////////////////////////////////////////////////////////////////////
int Hash512(int hashbitlen, const Byte * data, DataLength databitlen, Byte * hashval) {

	int ret;
	hashState512 state;

	if (data == NULL && databitlen > 0){
		return FAIL;
	}
	if (hashval == NULL){
		return FAIL;
	}

	ret = Init512(&state, hashbitlen);
	if (ret != SUCCESS) return ret;

	Update512(&state, data, databitlen);	

	Final512(&state, hashval);

	return SUCCESS;
}

