#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "lsh256_ref.h"

void compress256(hashState256 * state, const Byte * datablock) {
	int j, l, k;
	u32 vl, vr;

	/*expanded message*/
	u32 m[16 * (NS256 + 1)];

	u32 T[16];

	//message expansion
	for (l = 0; l < 32; l++){
		m[l] = U8TO32_LE(datablock + 4 * l);
	}

	for (j = 2; j <= NS256; j++){
		k = 16*j;
		m[k +  0] = m[k - 16] + m[k - 29];
		m[k +  1] = m[k - 15] + m[k - 30];
		m[k +  2] = m[k - 14] + m[k - 32];
		m[k +  3] = m[k - 13] + m[k - 31];
		m[k +  4] = m[k - 12] + m[k - 25];
		m[k +  5] = m[k - 11] + m[k - 28];
		m[k +  6] = m[k - 10] + m[k - 27];
		m[k +  7] = m[k -  9] + m[k - 26];
		m[k +  8] = m[k -  8] + m[k - 21];
		m[k +  9] = m[k -  7] + m[k - 22];
		m[k + 10] = m[k -  6] + m[k - 24];
		m[k + 11] = m[k -  5] + m[k - 23];
		m[k + 12] = m[k -  4] + m[k - 17];
		m[k + 13] = m[k -  3] + m[k - 20];
		m[k + 14] = m[k -  2] + m[k - 19];
		m[k + 15] = m[k -  1] + m[k - 18];
	}

	for (j = 0; j < NS256/2; j++){

		//MsgAdd & Mix
		k = 2*j;
		for (l = 0; l < 8; l++){
			vl = state->cv256[l]^m[16*k + l];
			vr = state->cv256[l + 8]^m[16*k + l+8];
			vl = ROL32(vl + vr, 29) ^ SC256[k][l];
			vr = ROL32(vr + vl, 1);
			T[l] = vr + vl; 
			T[l + 8] = ROL32(vr, gamma256[l]);
		}
		//WordPerm
		state->cv256[0] = T[ 6];state->cv256[ 8] = T[ 2];
		state->cv256[1] = T[ 4];state->cv256[ 9] = T[ 0];
		state->cv256[2] = T[ 5];state->cv256[10] = T[ 1];
		state->cv256[3] = T[ 7];state->cv256[11] = T[ 3];
		state->cv256[4] = T[12];state->cv256[12] = T[ 8];
		state->cv256[5] = T[15];state->cv256[13] = T[11];
		state->cv256[6] = T[14];state->cv256[14] = T[10];
		state->cv256[7] = T[13];state->cv256[15] = T[ 9];


		//MsgAdd & Mix
		k = 2*j+1;
		for (l = 0; l < 8; l++){
			vl = state->cv256[l]^m[16*k + l];
			vr = state->cv256[l + 8]^m[16*k + l+8];	
			vl = ROL32(vl + vr, 5)^ SC256[k][l];
			vr = ROL32(vl + vr, 17);
			T[l] = vr + vl; 
			T[l + 8] = ROL32(vr, gamma256[l]);
		}
		//WordPerm
		state->cv256[0] = T[ 6];state->cv256[ 8] = T[ 2];
		state->cv256[1] = T[ 4];state->cv256[ 9] = T[ 0];
		state->cv256[2] = T[ 5];state->cv256[10] = T[ 1];
		state->cv256[3] = T[ 7];state->cv256[11] = T[ 3];
		state->cv256[4] = T[12];state->cv256[12] = T[ 8];
		state->cv256[5] = T[15];state->cv256[13] = T[11];
		state->cv256[6] = T[14];state->cv256[14] = T[10];
		state->cv256[7] = T[13];state->cv256[15] = T[ 9];
	}

	for (l = 0; l < 16; l++) state->cv256[l] ^= m[16 * NS256 + l];

	return;
}
int Init256(hashState256 * state, int hashbitlen) {
	int j, l;
	int k;
	u32 vl, vr;

	u32 T[16];

	if ((hashbitlen <0) || (hashbitlen>256))
		return FAIL;
	else if (hashbitlen < 256){
		memset(state->cv256, 0, 16 * sizeof(u32));
		state->cv256[0] = 32;
		state->cv256[1] = (u32)hashbitlen;
		
		for (j = 0; j < NS256/2; j++)
		{
			//Mix
			k = 2*j;
			for (l = 0; l < 8; l++){
				vl = state->cv256[l];
				vr = state->cv256[l + 8];
				vl = ROL32(vl + vr, 29) ^ SC256[k][l];
				vr = ROL32(vr + vl, 1);
				T[l] = vr + vl; 
				T[l + 8] = ROL32(vr, gamma256[l]);
			}
			//WordPerm
			state->cv256[0] = T[ 6];state->cv256[ 8] = T[ 2];
			state->cv256[1] = T[ 4];state->cv256[ 9] = T[ 0];
			state->cv256[2] = T[ 5];state->cv256[10] = T[ 1];
			state->cv256[3] = T[ 7];state->cv256[11] = T[ 3];
			state->cv256[4] = T[12];state->cv256[12] = T[ 8];
			state->cv256[5] = T[15];state->cv256[13] = T[11];
			state->cv256[6] = T[14];state->cv256[14] = T[10];
			state->cv256[7] = T[13];state->cv256[15] = T[ 9];
			
			//Mix
			k = 2*j+1;
			for (l = 0; l < 8; l++){
				vl = state->cv256[l];
				vr = state->cv256[l + 8];
				vl = ROL32(vl + vr, 5)^ SC256[k][l];
				vr = ROL32(vl + vr, 17);
				T[l] = vr + vl; 
				T[l + 8] = ROL32(vr, gamma256[l]);
			}
			//WordPerm
			state->cv256[0] = T[ 6];state->cv256[ 8] = T[ 2];
			state->cv256[1] = T[ 4];state->cv256[ 9] = T[ 0];
			state->cv256[2] = T[ 5];state->cv256[10] = T[ 1];
			state->cv256[3] = T[ 7];state->cv256[11] = T[ 3];
			state->cv256[4] = T[12];state->cv256[12] = T[ 8];
			state->cv256[5] = T[15];state->cv256[13] = T[11];
			state->cv256[6] = T[14];state->cv256[14] = T[10];
			state->cv256[7] = T[13];state->cv256[15] = T[ 9];
		}
	}
	else /* if hashbitlen == 256*/
	{
		memcpy(state->cv256, IV256, sizeof(IV256));
	}

	state->hashbitlen = hashbitlen;
	return SUCCESS;
}
void Update256(hashState256 * state, const Byte * data, DataLength databitlen) {

	u64 numBlocks, temp;
	u32 pos1, pos2;
	int i;
	numBlocks = ((u64)databitlen >> 10) + 1;	

	for (i = 0; i < numBlocks - 1; i++){
		compress256(state, data);
		data += 128;
	}

	//computation of the state->Last256 block (padded)
	//if databitlen not multiple of 1024
	/*
	databitlen = 1024*(numBlocks-1) + 8*pos1 + pos2,
	0<=pos1<128, 0<=pos2<7
	*/
	if ((u32)(databitlen & 0x3ff)){
		temp = (numBlocks - 1) << 7; //temp = 128*(numBlocks-1)
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

	for (l = 0; l < 8; l++) H[l] = (state->cv256[l]) ^ (state->cv256[l + 8]);

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

