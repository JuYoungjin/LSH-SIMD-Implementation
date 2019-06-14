#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "lsh512_ref_SSE_OPT.h"
#include <emmintrin.h>

#define TwoSTEPs512(j) \
	TT[0] = _mm_xor_si128( TT[0], M[0]);\
	TT[1] = _mm_xor_si128( TT[1], M[1]);\
	TT[2] = _mm_xor_si128( TT[2], M[2]);\
	TT[3] = _mm_xor_si128( TT[3], M[3]);\
	TT[4] = _mm_xor_si128( TT[4], M[4]);\
	TT[5] = _mm_xor_si128( TT[5], M[5]);\
	TT[6] = _mm_xor_si128( TT[6], M[6]);\
	TT[7] = _mm_xor_si128( TT[7], M[7]);\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 23), _mm_srli_epi64( TT[0], 41));\
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[j][0])));\
	\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 23), _mm_srli_epi64( TT[1], 41));\
	TT[1] = _mm_xor_si128( TT[1],_mm_load_si128((__m128i*)(&SC512[j][2])));\
	\
	TT[2] = _mm_add_epi64( TT[2], TT[6]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 23), _mm_srli_epi64( TT[2], 41));\
	TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[j][4])));\
	\
	TT[3] = _mm_add_epi64( TT[3], TT[7]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 23), _mm_srli_epi64( TT[3], 41));\
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[j][6])));\
	\
	TT[4] = _mm_add_epi64( TT[4], TT[0]);\
	TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 59), _mm_srli_epi64( TT[4], 5));\
	TT[5] = _mm_add_epi64( TT[5], TT[1]);\
	TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 59), _mm_srli_epi64( TT[5], 5));\
	TT[6] = _mm_add_epi64( TT[6], TT[2]);\
	TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 59), _mm_srli_epi64( TT[6], 5));\
	TT[7] = _mm_add_epi64( TT[7], TT[3]);\
	TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 59), _mm_srli_epi64( TT[7], 5));\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	\
	Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);\
	Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);\
	\
	TT[0] = _mm_add_epi64( TT[2], TT[6]);\
	TT[1] = _mm_add_epi64( TT[3], TT[7]);\
	\
	TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));\
	TT[2] = _mm_shufflelo_epi16(TT[6],78);\
	\
	TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));\
	TT[3] = _mm_shuffle_epi32(TT[7],75);\
	\
	TT[4] = _mm_shufflehi_epi16(TT[4],147);\
	TT[5] = _mm_shufflehi_epi16(TT[5],147);\
	TT[5] = _mm_shuffle_epi32(TT[5],177);\
	TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);\
	TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);\
	\
	TT[4] = Temp;\
	TT[5] = Temp2;\
	\
	\
	Temp = _mm_shuffle_epi32( M[1], 78);\
	M[1] = _mm_add_epi64(M[0], M[9]);\
	M[0] = _mm_add_epi64(Temp, M[8]);\
	\
	Temp = _mm_shuffle_epi32( M[2], 78);\
	M[2] = _mm_add_epi64(M[3], M[10]);\
	M[3] = _mm_add_epi64(Temp, M[11]);\
	\
	Temp = _mm_shuffle_epi32( M[5], 78);\
	M[5] = _mm_add_epi64(M[4], M[13]);\
	M[4] = _mm_add_epi64(Temp, M[12]);\
	\
	Temp = _mm_shuffle_epi32( M[6], 78);\
	M[6] = _mm_add_epi64(M[7], M[14]);\
	M[7] = _mm_add_epi64(Temp, M[15]);\
	\
	\
	TT[0] = _mm_xor_si128( TT[0], M[8]);\
	TT[1] = _mm_xor_si128( TT[1], M[9]);\
	TT[2] = _mm_xor_si128( TT[2], M[10]);\
	TT[3] = _mm_xor_si128( TT[3], M[11]);\
	TT[4] = _mm_xor_si128( TT[4], M[12]);\
	TT[5] = _mm_xor_si128( TT[5], M[13]);\
	TT[6] = _mm_xor_si128( TT[6], M[14]);\
	TT[7] = _mm_xor_si128( TT[7], M[15]);\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 7), _mm_srli_epi64( TT[0], 57));\
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[j+1][0])));\
	\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 7), _mm_srli_epi64( TT[1], 57));\
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC512[j+1][2])));\
	\
	TT[2] = _mm_add_epi64( TT[2], TT[6]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 7), _mm_srli_epi64( TT[2], 57));\
	TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[j+1][4])));\
	\
	TT[3] = _mm_add_epi64( TT[3], TT[7]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 7), _mm_srli_epi64( TT[3], 57));\
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[j+1][6])));\
	\
	TT[4] = _mm_add_epi64( TT[4], TT[0]);\
	TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 3), _mm_srli_epi64( TT[4], 61));\
	TT[5] = _mm_add_epi64( TT[5], TT[1]);\
	TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 3), _mm_srli_epi64( TT[5], 61));\
	TT[6] = _mm_add_epi64( TT[6], TT[2]);\
	TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 3), _mm_srli_epi64( TT[6], 61));\
	TT[7] = _mm_add_epi64( TT[7], TT[3]);\
	TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 3), _mm_srli_epi64( TT[7], 61));\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	\
	Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);\
	Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);\
	\
	TT[0] = _mm_add_epi64( TT[2], TT[6]);\
	TT[1] = _mm_add_epi64( TT[3], TT[7]);\
	\
	TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));\
	TT[2] = _mm_shufflelo_epi16(TT[6],78);\
	\
	TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));\
	TT[3] = _mm_shuffle_epi32(TT[7],75);\
	\
	TT[4] = _mm_shufflehi_epi16(TT[4],147);\
	TT[5] = _mm_shufflehi_epi16(TT[5],147);\
	TT[5] = _mm_shuffle_epi32(TT[5],177);\
	TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);\
	TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);\
	\
	TT[4] = Temp;\
	TT[5] = Temp2;\
	\
	Temp = _mm_shuffle_epi32( M[9], 78);\
	M[9] = _mm_add_epi64(M[8], M[1]);\
	M[8] = _mm_add_epi64(Temp, M[0]);\
	\
	Temp = _mm_shuffle_epi32( M[10], 78);\
	M[10] = _mm_add_epi64(M[11], M[2]);\
	M[11] = _mm_add_epi64(Temp, M[3]);\
	\
	Temp = _mm_shuffle_epi32( M[13], 78);\
	M[13] = _mm_add_epi64(M[12], M[5]);\
	M[12] = _mm_add_epi64(Temp, M[4]);\
	\
	Temp = _mm_shuffle_epi32( M[14], 78);\
	M[14] = _mm_add_epi64(M[15], M[6]);\
	M[15] = _mm_add_epi64(Temp, M[7]);\

#define TwoSTEPs512_wo_ME(j) \
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 23), _mm_srli_epi64( TT[0], 41));\
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[j][0])));\
	\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 23), _mm_srli_epi64( TT[1], 41));\
	TT[1] = _mm_xor_si128( TT[1],_mm_load_si128((__m128i*)(&SC512[j][2])));\
	\
	TT[2] = _mm_add_epi64( TT[2], TT[6]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 23), _mm_srli_epi64( TT[2], 41));\
	TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[j][4])));\
	\
	TT[3] = _mm_add_epi64( TT[3], TT[7]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 23), _mm_srli_epi64( TT[3], 41));\
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[j][6])));\
	\
	TT[4] = _mm_add_epi64( TT[4], TT[0]);\
	TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 59), _mm_srli_epi64( TT[4], 5));\
	TT[5] = _mm_add_epi64( TT[5], TT[1]);\
	TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 59), _mm_srli_epi64( TT[5], 5));\
	TT[6] = _mm_add_epi64( TT[6], TT[2]);\
	TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 59), _mm_srli_epi64( TT[6], 5));\
	TT[7] = _mm_add_epi64( TT[7], TT[3]);\
	TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 59), _mm_srli_epi64( TT[7], 5));\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	\
	Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);\
	Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);\
	\
	TT[0] = _mm_add_epi64( TT[2], TT[6]);\
	TT[1] = _mm_add_epi64( TT[3], TT[7]);\
	\
	TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));\
	TT[2] = _mm_shufflelo_epi16(TT[6],78);\
	\
	TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));\
	TT[3] = _mm_shuffle_epi32(TT[7],75);\
	\
	TT[4] = _mm_shufflehi_epi16(TT[4],147);\
	TT[5] = _mm_shufflehi_epi16(TT[5],147);\
	TT[5] = _mm_shuffle_epi32(TT[5],177);\
	TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);\
	TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);\
	\
	TT[4] = Temp;\
	TT[5] = Temp2;\
	\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 7), _mm_srli_epi64( TT[0], 57));\
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[j+1][0])));\
	\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 7), _mm_srli_epi64( TT[1], 57));\
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC512[j+1][2])));\
	\
	TT[2] = _mm_add_epi64( TT[2], TT[6]);\
	TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 7), _mm_srli_epi64( TT[2], 57));\
	TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[j+1][4])));\
	\
	TT[3] = _mm_add_epi64( TT[3], TT[7]);\
	TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 7), _mm_srli_epi64( TT[3], 57));\
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[j+1][6])));\
	\
	TT[4] = _mm_add_epi64( TT[4], TT[0]);\
	TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 3), _mm_srli_epi64( TT[4], 61));\
	TT[5] = _mm_add_epi64( TT[5], TT[1]);\
	TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 3), _mm_srli_epi64( TT[5], 61));\
	TT[6] = _mm_add_epi64( TT[6], TT[2]);\
	TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 3), _mm_srli_epi64( TT[6], 61));\
	TT[7] = _mm_add_epi64( TT[7], TT[3]);\
	TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 3), _mm_srli_epi64( TT[7], 61));\
	\
	TT[0] = _mm_add_epi64( TT[0], TT[4]);\
	TT[1] = _mm_add_epi64( TT[1], TT[5]);\
	\
	Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);\
	Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);\
	\
	TT[0] = _mm_add_epi64( TT[2], TT[6]);\
	TT[1] = _mm_add_epi64( TT[3], TT[7]);\
	\
	TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));\
	TT[2] = _mm_shufflelo_epi16(TT[6],78);\
	\
	TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));\
	TT[3] = _mm_shuffle_epi32(TT[7],75);\
	\
	TT[4] = _mm_shufflehi_epi16(TT[4],147);\
	TT[5] = _mm_shufflehi_epi16(TT[5],147);\
	TT[5] = _mm_shuffle_epi32(TT[5],177);\
	TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);\
	TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);\
	\
	TT[4] = Temp;\
	TT[5] = Temp2;\


void compress512(hashState512 * state, const Byte * datablock) {
	int j;
	ALIGN(16) u64 m[32];
	ALIGN(16) const u64 Mask1[2] = {0xFFFFFFFFFFFFFFFFULL,0};
	ALIGN(16) const u64 Mask2[2] = {0,0xFFFFFFFFFFFFFFFFULL};

	__m128i M1 = _mm_load_si128((__m128i*)Mask1);
	__m128i M2 = _mm_load_si128((__m128i*)Mask2);
	__m128i Temp;
	__m128i Temp2;
	__m128i M[16];
	__m128i TT[8];

	m[ 0] = U8TO64_LE(datablock      );		m[16] = U8TO64_LE(datablock       + 128);
	m[ 1] = U8TO64_LE(datablock +   8);		m[17] = U8TO64_LE(datablock +   8 + 128);
	m[ 2] = U8TO64_LE(datablock +  16);		m[18] = U8TO64_LE(datablock +  16 + 128);
	m[ 3] = U8TO64_LE(datablock +  24);		m[19] = U8TO64_LE(datablock +  24 + 128);
	m[ 4] = U8TO64_LE(datablock +  48);		m[20] = U8TO64_LE(datablock +  48 + 128);
	m[ 5] = U8TO64_LE(datablock +  32);		m[21] = U8TO64_LE(datablock +  32 + 128);
	m[ 6] = U8TO64_LE(datablock +  40);		m[22] = U8TO64_LE(datablock +  40 + 128);
	m[ 7] = U8TO64_LE(datablock +  56);		m[23] = U8TO64_LE(datablock +  56 + 128);
	m[ 8] = U8TO64_LE(datablock +  64);		m[24] = U8TO64_LE(datablock +  64 + 128);
	m[ 9] = U8TO64_LE(datablock +  72);		m[25] = U8TO64_LE(datablock +  72 + 128);
	m[10] = U8TO64_LE(datablock +  80);		m[26] = U8TO64_LE(datablock +  80 + 128);
	m[11] = U8TO64_LE(datablock +  88);		m[27] = U8TO64_LE(datablock +  88 + 128);
	m[12] = U8TO64_LE(datablock + 112);		m[28] = U8TO64_LE(datablock + 112 + 128);
	m[13] = U8TO64_LE(datablock +  96);		m[29] = U8TO64_LE(datablock +  96 + 128);
	m[14] = U8TO64_LE(datablock + 104);		m[30] = U8TO64_LE(datablock + 104 + 128);
	m[15] = U8TO64_LE(datablock + 120);		m[31] = U8TO64_LE(datablock + 120 + 128);

	M[0] = _mm_load_si128((__m128i*)(m));
	M[1] = _mm_load_si128((__m128i*)(m+2));
	M[2] = _mm_load_si128((__m128i*)(m+4));
	M[3] = _mm_load_si128((__m128i*)(m+6));
	M[4] = _mm_load_si128((__m128i*)(m+8));
	M[5] = _mm_load_si128((__m128i*)(m+10));
	M[6] = _mm_load_si128((__m128i*)(m+12));
	M[7] = _mm_load_si128((__m128i*)(m+14));
	M[8] = _mm_load_si128((__m128i*)(m+16));
	M[9] = _mm_load_si128((__m128i*)(m+18));
	M[10] = _mm_load_si128((__m128i*)(m+20));
	M[11] = _mm_load_si128((__m128i*)(m+22));
	M[12] = _mm_load_si128((__m128i*)(m+24));
	M[13] = _mm_load_si128((__m128i*)(m+26));
	M[14] = _mm_load_si128((__m128i*)(m+28));
	M[15] = _mm_load_si128((__m128i*)(m+30));

	TT[0] = _mm_load_si128((__m128i*)state->cv512);
	TT[1] = _mm_load_si128((__m128i*)(state->cv512+2));
	TT[2] = _mm_load_si128((__m128i*)(state->cv512+4));
	TT[3] = _mm_load_si128((__m128i*)(state->cv512+6));
	TT[4] = _mm_load_si128((__m128i*)(state->cv512+8));
	TT[5] = _mm_load_si128((__m128i*)(state->cv512+10));
	TT[6] = _mm_load_si128((__m128i*)(state->cv512+12));
	TT[7] = _mm_load_si128((__m128i*)(state->cv512+14));
	
	
	for(j=0;j<26;j+=2)
	{
		TwoSTEPs512(j);
	}
	

	//26 STEP start
	TT[0] = _mm_xor_si128( TT[0], M[0]);
	TT[1] = _mm_xor_si128( TT[1], M[1]);
	TT[2] = _mm_xor_si128( TT[2], M[2]);
	TT[3] = _mm_xor_si128( TT[3], M[3]);
	TT[4] = _mm_xor_si128( TT[4], M[4]);
	TT[5] = _mm_xor_si128( TT[5], M[5]);
	TT[6] = _mm_xor_si128( TT[6], M[6]);
	TT[7] = _mm_xor_si128( TT[7], M[7]);
	
	TT[0] = _mm_add_epi64( TT[0], TT[4]);
	TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 23), _mm_srli_epi64( TT[0], 41));
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[26][0])));
	
	TT[1] = _mm_add_epi64( TT[1], TT[5]);
	TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 23), _mm_srli_epi64( TT[1], 41));
	TT[1] = _mm_xor_si128( TT[1],_mm_load_si128((__m128i*)(&SC512[26][2])));
	
	TT[2] = _mm_add_epi64( TT[2], TT[6]);
	TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 23), _mm_srli_epi64( TT[2], 41));
	TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[26][4])));
	
	TT[3] = _mm_add_epi64( TT[3], TT[7]);
	TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 23), _mm_srli_epi64( TT[3], 41));
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[26][6])));
	
	TT[4] = _mm_add_epi64( TT[4], TT[0]);
	TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 59), _mm_srli_epi64( TT[4], 5));
	TT[5] = _mm_add_epi64( TT[5], TT[1]);
	TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 59), _mm_srli_epi64( TT[5], 5));
	TT[6] = _mm_add_epi64( TT[6], TT[2]);
	TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 59), _mm_srli_epi64( TT[6], 5));
	TT[7] = _mm_add_epi64( TT[7], TT[3]);
	TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 59), _mm_srli_epi64( TT[7], 5));
	
	TT[0] = _mm_add_epi64( TT[0], TT[4]);
	TT[1] = _mm_add_epi64( TT[1], TT[5]);
	
	Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);
	Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);
	
	TT[0] = _mm_add_epi64( TT[2], TT[6]);
	TT[1] = _mm_add_epi64( TT[3], TT[7]);
	
	TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));
	TT[2] = _mm_shufflelo_epi16(TT[6],78);
	
	TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));
	TT[3] = _mm_shuffle_epi32(TT[7],75);
	
	TT[4] = _mm_shufflehi_epi16(TT[4],147);
	TT[5] = _mm_shufflehi_epi16(TT[5],147);
	TT[5] = _mm_shuffle_epi32(TT[5],177);
	TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);
	TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);
	
	TT[4] = Temp;
	TT[5] = Temp2;
	
	
	Temp = _mm_shuffle_epi32( M[1], 78);
	M[1] = _mm_add_epi64(M[0], M[9]);
	M[0] = _mm_add_epi64(Temp, M[8]);
	
	Temp = _mm_shuffle_epi32( M[2], 78);
	M[2] = _mm_add_epi64(M[3], M[10]);
	M[3] = _mm_add_epi64(Temp, M[11]);
	
	Temp = _mm_shuffle_epi32( M[5], 78);
	M[5] = _mm_add_epi64(M[4], M[13]);
	M[4] = _mm_add_epi64(Temp, M[12]);
	
	Temp = _mm_shuffle_epi32( M[6], 78);
	M[6] = _mm_add_epi64(M[7], M[14]);
	M[7] = _mm_add_epi64(Temp, M[15]);
	
	//27 STEP start	
	TT[0] = _mm_xor_si128( TT[0], M[8]);
	TT[1] = _mm_xor_si128( TT[1], M[9]);
	TT[2] = _mm_xor_si128( TT[2], M[10]);
	TT[3] = _mm_xor_si128( TT[3], M[11]);
	TT[4] = _mm_xor_si128( TT[4], M[12]);
	TT[5] = _mm_xor_si128( TT[5], M[13]);
	TT[6] = _mm_xor_si128( TT[6], M[14]);
	TT[7] = _mm_xor_si128( TT[7], M[15]);
	
	TT[0] = _mm_add_epi64( TT[0], TT[4]);
	TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 7), _mm_srli_epi64( TT[0], 57));
	TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[27][0])));
	
	TT[1] = _mm_add_epi64( TT[1], TT[5]);
	TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 7), _mm_srli_epi64( TT[1], 57));
	TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC512[27][2])));
	
	TT[2] = _mm_add_epi64( TT[2], TT[6]);
	TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 7), _mm_srli_epi64( TT[2], 57));
	TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[27][4])));
	
	TT[3] = _mm_add_epi64( TT[3], TT[7]);
	TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 7), _mm_srli_epi64( TT[3], 57));
	TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[27][6])));
	
	TT[4] = _mm_add_epi64( TT[4], TT[0]);
	TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 3), _mm_srli_epi64( TT[4], 61));
	TT[5] = _mm_add_epi64( TT[5], TT[1]);
	TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 3), _mm_srli_epi64( TT[5], 61));
	TT[6] = _mm_add_epi64( TT[6], TT[2]);
	TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 3), _mm_srli_epi64( TT[6], 61));
	TT[7] = _mm_add_epi64( TT[7], TT[3]);
	TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 3), _mm_srli_epi64( TT[7], 61));
	
	TT[0] = _mm_add_epi64( TT[0], TT[4]);
	TT[1] = _mm_add_epi64( TT[1], TT[5]);
	
	Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);
	Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);
	
	TT[0] = _mm_add_epi64( TT[2], TT[6]);
	TT[1] = _mm_add_epi64( TT[3], TT[7]);
	
	TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));
	TT[2] = _mm_shufflelo_epi16(TT[6],78);
	
	TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));
	TT[3] = _mm_shuffle_epi32(TT[7],75);
	
	TT[4] = _mm_shufflehi_epi16(TT[4],147);
	TT[5] = _mm_shufflehi_epi16(TT[5],147);
	TT[5] = _mm_shuffle_epi32(TT[5],177);
	TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);
	TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);

	TT[0] = _mm_xor_si128( TT[0], M[0]);
	TT[1] = _mm_xor_si128( TT[1], M[1]);
	TT[2] = _mm_xor_si128( TT[2], M[2]);
	TT[3] = _mm_xor_si128( TT[3], M[3]);
	TT[4] = _mm_xor_si128( Temp, M[4]);
	TT[5] = _mm_xor_si128( Temp2, M[5]);
	TT[6] = _mm_xor_si128( TT[6], M[6]);
	TT[7] = _mm_xor_si128( TT[7], M[7]);
	
	_mm_storeu_si128((__m128i*)(&state->cv512[0]),TT[0]);
	_mm_storeu_si128((__m128i*)(&state->cv512[2]),TT[1]);
	_mm_storeu_si128((__m128i*)(&state->cv512[4]),TT[2]);
	_mm_storeu_si128((__m128i*)(&state->cv512[6]),TT[3]);
	_mm_storeu_si128((__m128i*)(&state->cv512[8]),TT[4]);
	_mm_storeu_si128((__m128i*)(&state->cv512[10]),TT[5]);
	_mm_storeu_si128((__m128i*)(&state->cv512[12]),TT[6]);
	_mm_storeu_si128((__m128i*)(&state->cv512[14]),TT[7]);

	return;
}
int Init512(hashState512 * state, int hashbitlen) {

	int j;
	
	if(hashbitlen == 512) /* if hashbitlen == 512 */
	{
		memcpy(state->cv512, IV512, sizeof(IV512));
	}
	else if ((hashbitlen <0) || (hashbitlen>512))
		return FAIL;
	else{
		memset(state->cv512, 0, 16 * sizeof(u64));
		state->cv512[0] = 64;
		state->cv512[1] = (u64)hashbitlen;
		
		ALIGN(16) u64 m[32];
		ALIGN(16) const u64 Mask1[2] = {0xFFFFFFFFFFFFFFFFULL,0};
		ALIGN(16) const u64 Mask2[2] = {0,0xFFFFFFFFFFFFFFFFULL};

		__m128i M1 = _mm_load_si128((__m128i*)Mask1);
		__m128i M2 = _mm_load_si128((__m128i*)Mask2);
		__m128i Temp;
		__m128i Temp2;
		__m128i TT[8];

		TT[0] = _mm_load_si128((__m128i*)state->cv512);
		TT[1] = _mm_load_si128((__m128i*)(state->cv512+2));
		TT[2] = _mm_load_si128((__m128i*)(state->cv512+4));
		TT[3] = _mm_load_si128((__m128i*)(state->cv512+6));
		TT[4] = _mm_load_si128((__m128i*)(state->cv512+8));
		TT[5] = _mm_load_si128((__m128i*)(state->cv512+10));
		TT[6] = _mm_load_si128((__m128i*)(state->cv512+12));
		TT[7] = _mm_load_si128((__m128i*)(state->cv512+14));
		
		
		for(j=0;j<26;j+=2)
		{
			TwoSTEPs512_wo_ME(j);
		}
		

		//26 STEP start
		TT[0] = _mm_add_epi64( TT[0], TT[4]);
		TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 23), _mm_srli_epi64( TT[0], 41));
		TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[26][0])));
		
		TT[1] = _mm_add_epi64( TT[1], TT[5]);
		TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 23), _mm_srli_epi64( TT[1], 41));
		TT[1] = _mm_xor_si128( TT[1],_mm_load_si128((__m128i*)(&SC512[26][2])));
		
		TT[2] = _mm_add_epi64( TT[2], TT[6]);
		TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 23), _mm_srli_epi64( TT[2], 41));
		TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[26][4])));
		
		TT[3] = _mm_add_epi64( TT[3], TT[7]);
		TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 23), _mm_srli_epi64( TT[3], 41));
		TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[26][6])));
		
		TT[4] = _mm_add_epi64( TT[4], TT[0]);
		TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 59), _mm_srli_epi64( TT[4], 5));
		TT[5] = _mm_add_epi64( TT[5], TT[1]);
		TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 59), _mm_srli_epi64( TT[5], 5));
		TT[6] = _mm_add_epi64( TT[6], TT[2]);
		TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 59), _mm_srli_epi64( TT[6], 5));
		TT[7] = _mm_add_epi64( TT[7], TT[3]);
		TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 59), _mm_srli_epi64( TT[7], 5));
		
		TT[0] = _mm_add_epi64( TT[0], TT[4]);
		TT[1] = _mm_add_epi64( TT[1], TT[5]);
		
		Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);
		Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);
		
		TT[0] = _mm_add_epi64( TT[2], TT[6]);
		TT[1] = _mm_add_epi64( TT[3], TT[7]);
		
		TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));
		TT[2] = _mm_shufflelo_epi16(TT[6],78);
		
		TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));
		TT[3] = _mm_shuffle_epi32(TT[7],75);
		
		TT[4] = _mm_shufflehi_epi16(TT[4],147);
		TT[5] = _mm_shufflehi_epi16(TT[5],147);
		TT[5] = _mm_shuffle_epi32(TT[5],177);
		TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);
		TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);
		
		TT[4] = Temp;
		TT[5] = Temp2;
					
		//27 STEP start	
		TT[0] = _mm_add_epi64( TT[0], TT[4]);
		TT[0] = _mm_xor_si128(_mm_slli_epi64( TT[0], 7), _mm_srli_epi64( TT[0], 57));
		TT[0] = _mm_xor_si128( TT[0], _mm_load_si128((__m128i*)(&SC512[27][0])));
		
		TT[1] = _mm_add_epi64( TT[1], TT[5]);
		TT[1] = _mm_xor_si128(_mm_slli_epi64( TT[1], 7), _mm_srli_epi64( TT[1], 57));
		TT[1] = _mm_xor_si128( TT[1], _mm_load_si128((__m128i*)(&SC512[27][2])));
		
		TT[2] = _mm_add_epi64( TT[2], TT[6]);
		TT[2] = _mm_xor_si128(_mm_slli_epi64( TT[2], 7), _mm_srli_epi64( TT[2], 57));
		TT[2] = _mm_xor_si128( TT[2], _mm_load_si128((__m128i*)(&SC512[27][4])));
		
		TT[3] = _mm_add_epi64( TT[3], TT[7]);
		TT[3] = _mm_xor_si128(_mm_slli_epi64( TT[3], 7), _mm_srli_epi64( TT[3], 57));
		TT[3] = _mm_xor_si128( TT[3], _mm_load_si128((__m128i*)(&SC512[27][6])));
		
		TT[4] = _mm_add_epi64( TT[4], TT[0]);
		TT[4] = _mm_xor_si128(_mm_slli_epi64( TT[4], 3), _mm_srli_epi64( TT[4], 61));
		TT[5] = _mm_add_epi64( TT[5], TT[1]);
		TT[5] = _mm_xor_si128(_mm_slli_epi64( TT[5], 3), _mm_srli_epi64( TT[5], 61));
		TT[6] = _mm_add_epi64( TT[6], TT[2]);
		TT[6] = _mm_xor_si128(_mm_slli_epi64( TT[6], 3), _mm_srli_epi64( TT[6], 61));
		TT[7] = _mm_add_epi64( TT[7], TT[3]);
		TT[7] = _mm_xor_si128(_mm_slli_epi64( TT[7], 3), _mm_srli_epi64( TT[7], 61));
		
		TT[0] = _mm_add_epi64( TT[0], TT[4]);
		TT[1] = _mm_add_epi64( TT[1], TT[5]);
		
		Temp  = _mm_unpacklo_epi64(TT[1],TT[0]);
		Temp2 = _mm_unpackhi_epi64(TT[0],TT[1]);
		
		TT[0] = _mm_add_epi64( TT[2], TT[6]);
		TT[1] = _mm_add_epi64( TT[3], TT[7]);
		
		TT[6] = _mm_xor_si128(_mm_slli_epi64(TT[6], 8), _mm_srli_epi64(TT[6], 56));
		TT[2] = _mm_shufflelo_epi16(TT[6],78);
		
		TT[7] = _mm_xor_si128(_mm_slli_epi64(TT[7], 24), _mm_srli_epi64(TT[7], 40));
		TT[3] = _mm_shuffle_epi32(TT[7],75);
		
		TT[4] = _mm_shufflehi_epi16(TT[4],147);
		TT[5] = _mm_shufflehi_epi16(TT[5],147);
		TT[5] = _mm_shuffle_epi32(TT[5],177);
		TT[6] = _mm_unpacklo_epi64(TT[5],TT[4]);
		TT[7] = _mm_unpackhi_epi64(TT[5],TT[4]);
		
		_mm_storeu_si128((__m128i*)(&state->cv512[0]),TT[0]);
		_mm_storeu_si128((__m128i*)(&state->cv512[2]),TT[1]);
		_mm_storeu_si128((__m128i*)(&state->cv512[4]),TT[2]);
		_mm_storeu_si128((__m128i*)(&state->cv512[6]),TT[3]);
		_mm_storeu_si128((__m128i*)(&state->cv512[8]),Temp);
		_mm_storeu_si128((__m128i*)(&state->cv512[10]),Temp2);
		_mm_storeu_si128((__m128i*)(&state->cv512[12]),TT[6]);
		_mm_storeu_si128((__m128i*)(&state->cv512[14]),TT[7]);
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

