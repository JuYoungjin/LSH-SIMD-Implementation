#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "lsh256_ref_AVX2_OPT.h"
#include <immintrin.h>

#define TwoSTEPs256(j) \
	TT[1] = _mm256_xor_si256( TT[1], M[1]);\
	\
	TT[0] = _mm256_add_epi32( _mm256_xor_si256( Temp, M[0]), TT[1]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 29), _mm256_srli_epi32( TT[0], 3));\
	\
	TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[j][0])));\
	\
	TT[1] = _mm256_add_epi32( TT[1], TT[0]);\
	\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 1), _mm256_srli_epi32( TT[1], 31));\
	\
	TT[0] = _mm256_shuffle_epi8(_mm256_add_epi32( TT[0], TT[1]),P);\
	TT[1] = _mm256_shuffle_epi8(TT[1],G);\
	\
	Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);\
	TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);\
	\
	\
	TT[1] = _mm256_xor_si256( TT[1], M[3]);\
	\
	TT[0] = _mm256_add_epi32( _mm256_xor_si256( Temp, M[2]), TT[1]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 5), _mm256_srli_epi32( TT[0], 27));\
	\
	TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[j+1][0])));\
	\
	TT[1] = _mm256_add_epi32( TT[1], TT[0]);\
	\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 17), _mm256_srli_epi32( TT[1], 15));\
	\
	TT[0] = _mm256_shuffle_epi8(_mm256_add_epi32( TT[0], TT[1]),P);\
	TT[1] = _mm256_shuffle_epi8(TT[1],G);\
	\
	Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);\
	TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);\
	\
	M[0] = _mm256_shuffle_epi32( M[0], 30);\
	M[0] = _mm256_add_epi32(M[0], M[2]);\
	M[1] = _mm256_shuffle_epi32( M[1], 30);\
	M[1] = _mm256_add_epi32(M[1], M[3]);\
	M[2] = _mm256_shuffle_epi32( M[2], 30);\
	M[2] = _mm256_add_epi32(M[2], M[0]);\
	M[3] = _mm256_shuffle_epi32( M[3], 30);\
	M[3] = _mm256_add_epi32(M[3], M[1]);\

#define TwoSTEPs256_wo_ME(j) \
	TT[0] = _mm256_add_epi32( TT[0], TT[1]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 29), _mm256_srli_epi32( TT[0], 3));\
	\
	TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[j][0])));\
	\
	TT[1] = _mm256_add_epi32( TT[1], TT[0]);\
	\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 1), _mm256_srli_epi32( TT[1], 31));\
	\
	TT[0] = _mm256_shuffle_epi8(_mm256_add_epi32( TT[0], TT[1]),P);\
	TT[1] = _mm256_shuffle_epi8(TT[1],G);\
	\
	Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);\
	TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);\
	\
	\
	TT[0] = _mm256_add_epi32( Temp, TT[1]);\
	\
	TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 5), _mm256_srli_epi32( TT[0], 27));\
	\
	TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[j+1][0])));\
	\
	TT[1] = _mm256_add_epi32( TT[1], TT[0]);\
	\
	TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 17), _mm256_srli_epi32( TT[1], 15));\
	\
	TT[0] = _mm256_shuffle_epi8(_mm256_add_epi32( TT[0], TT[1]),P);\
	TT[1] = _mm256_shuffle_epi8(TT[1],G);\
	\
	Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);\
	TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);\
	TT[0] = Temp;\


void compress256(hashState256 * state, const Byte * datablock) {
	int i;
	ALIGN(32) u32 m[32];
	ALIGN(32) const u32 Gamma[8] = 
		{0x06050407,0x080b0a09,0x0d0c0f0e,0x03020100,0x05040706,0x03020100,0x0a09080b,0x0c0f0e0d};
	ALIGN(32) const u32 Perm[8] = 
		{0x0f0e0d0c,0x03020100,0x0b0a0908,0x07060504,0x0b0a0908,0x0f0e0d0c,0x03020100,0x07060504};
	__m256i G = _mm256_load_si256((__m256i*)Gamma);
	__m256i P = _mm256_load_si256((__m256i*)Perm);
	__m256i M[4];
	__m256i TT[2];
	__m256i Temp;
	

	for(i=0;i<32;i+=8)
	{ 
		m[i  ] = U8TO32_LE(datablock + 4*i     );
		m[i+1] = U8TO32_LE(datablock + 4*i +  4);
		m[i+2] = U8TO32_LE(datablock + 4*i + 12);
		m[i+3] = U8TO32_LE(datablock + 4*i +  8);
		m[i+4] = U8TO32_LE(datablock + 4*i + 28);
		m[i+5] = U8TO32_LE(datablock + 4*i + 20);
		m[i+6] = U8TO32_LE(datablock + 4*i + 24);
		m[i+7] = U8TO32_LE(datablock + 4*i + 16);
	}


	M[0] = _mm256_stream_load_si256((__m256i*)m);
	M[1] = _mm256_stream_load_si256((__m256i*)(m+8));
	M[2] = _mm256_stream_load_si256((__m256i*)(m+16));
	M[3] = _mm256_stream_load_si256((__m256i*)(m+24));

	Temp = _mm256_stream_load_si256((__m256i*)state->cv256);
	TT[1] = _mm256_stream_load_si256((__m256i*)(state->cv256+8));
	
	for(i=0;i<24;i+=4)
	{
		TwoSTEPs256(i);
		TwoSTEPs256(i+2);
	}

	//24 STEP start
	TT[1] = _mm256_xor_si256( TT[1], M[1]);
	
	TT[0] = _mm256_add_epi32( _mm256_xor_si256( Temp, M[0]), TT[1]);
	
	TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 29), _mm256_srli_epi32( TT[0], 3));
	
	TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[24][0])));
	
	TT[1] = _mm256_add_epi32( TT[1], TT[0]);
	
	TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 1), _mm256_srli_epi32( TT[1], 31));
	
	TT[0] = _mm256_shuffle_epi8(_mm256_add_epi32( TT[0], TT[1]),P);
	TT[1] = _mm256_shuffle_epi8(TT[1],G);
	
	Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);
	TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);
	
	M[0] = _mm256_shuffle_epi32( M[0], 30);
	M[0] = _mm256_add_epi32(M[0], M[2]);
	M[1] = _mm256_shuffle_epi32( M[1], 30);
	M[1] = _mm256_add_epi32(M[1], M[3]);
	
	//25 STEP start	
	TT[1] = _mm256_xor_si256( TT[1], M[3]);
	
	TT[0] = _mm256_add_epi32( _mm256_xor_si256( Temp, M[2]), TT[1]);
	
	TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 5), _mm256_srli_epi32( TT[0], 27));
	
	TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[25][0])));
	
	TT[1] = _mm256_add_epi32( TT[1], TT[0]);
	
	TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 17), _mm256_srli_epi32( TT[1], 15));
	
	TT[0] = _mm256_shuffle_epi8(_mm256_add_epi32( TT[0], TT[1]),P);
	TT[1] = _mm256_shuffle_epi8(TT[1],G);
	
	Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);
	TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);

	TT[0] = _mm256_xor_si256( Temp, M[0]);
	TT[1] = _mm256_xor_si256( TT[1], M[1]);

	//store
	_mm256_store_si256((__m256i*)(&state->cv256[0]),TT[0]);
	_mm256_store_si256((__m256i*)(&state->cv256[8]),TT[1]);
	return;
}
int Init256(hashState256 * state, int hashbitlen) {

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
		
		ALIGN(32) u32 m[16 * (NS256 + 1)];
		ALIGN(32) const u32 Gamma[8] = 
			{0x06050407,0x080b0a09,0x0d0c0f0e,0x03020100,0x05040706,0x03020100,0x0a09080b,0x0c0f0e0d};
		ALIGN(32) const u32 Perm[8] = 
			{0x0f0e0d0c,0x03020100,0x0b0a0908,0x07060504,0x0b0a0908,0x0f0e0d0c,0x03020100,0x07060504};
		__m256i G = _mm256_load_si256((__m256i*)Gamma);
		__m256i P = _mm256_load_si256((__m256i*)Perm);
		__m256i TT[2];
		__m256i Temp;
		
		TT[0] = _mm256_stream_load_si256((__m256i*)state->cv256);
		TT[1] = _mm256_stream_load_si256((__m256i*)(state->cv256+8));

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
		TT[0] = _mm256_add_epi32( TT[0], TT[1]);
		TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 29), _mm256_srli_epi32( TT[0], 3));
		TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[24][0])));
		TT[1] = _mm256_add_epi32( TT[1], TT[0]);
		TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 1), _mm256_srli_epi32( TT[1], 31));
		TT[0] = _mm256_add_epi32( TT[0], TT[1]);

		TT[0] = _mm256_shuffle_epi8(TT[0],P);
		TT[1] = _mm256_shuffle_epi8(TT[1],G);

		Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);
		TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);
		TT[0] = Temp;

		//25 STEP start	
		TT[0] = _mm256_add_epi32( TT[0], TT[1]);
		TT[0] = _mm256_xor_si256(_mm256_slli_epi32( TT[0], 5), _mm256_srli_epi32( TT[0], 27));
		TT[0] = _mm256_xor_si256( TT[0], _mm256_stream_load_si256((__m256i*)(&SC256[25][0])));
		TT[1] = _mm256_add_epi32( TT[1], TT[0]);
		TT[1] = _mm256_xor_si256(_mm256_slli_epi32( TT[1], 17), _mm256_srli_epi32( TT[1], 15));
		TT[0] = _mm256_add_epi32( TT[0], TT[1]);

		TT[0] = _mm256_shuffle_epi8(TT[0],P);
		TT[1] = _mm256_shuffle_epi8(TT[1],G);

		Temp  = _mm256_permute2x128_si256(TT[0],TT[1],0x31);
		TT[1] = _mm256_permute2x128_si256(TT[0],TT[1],0x20);

		//store
		_mm256_store_si256((__m256i*)(&state->cv256[0]),Temp);
		_mm256_store_si256((__m256i*)(&state->cv256[8]),TT[1]);
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
	H[2] = (state->cv256[3]) ^ (state->cv256[11]);
	H[3] = (state->cv256[2]) ^ (state->cv256[10]);
	H[4] = (state->cv256[7]) ^ (state->cv256[15]);
	H[5] = (state->cv256[5]) ^ (state->cv256[13]);
	H[6] = (state->cv256[6]) ^ (state->cv256[14]);
	H[7] = (state->cv256[4]) ^ (state->cv256[12]);

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

