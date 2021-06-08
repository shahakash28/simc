#include <emp-ot/emp-ot.h>
#include <emp-tool/emp-tool.h>
#include <iostream>
#include <iostream>
#include <emp-tool/utils/aes.h>
#include <emp-tool/utils/aes_opt.h>
#include <string.h>

using namespace emp;
using namespace std;

#if !defined (ALIGN16)
#if defined (__GNUC__)
#  define ALIGN16  __attribute__  ( (aligned (16)))
# else
#  define ALIGN16 __declspec (align (16))
# endif
#endif

ALIGN16 uint8_t AES256_TEST_KEY[] = {
	0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,
	0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,
	0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,
	0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4
};

ALIGN16 uint8_t AES256_EXPECTED[] = {
	0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,
	0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4,
	0x9b,0xa3,0x54,0x11,0x8e,0x69,0x25,0xaf,0xa5,0x1a,0x8b,0x5f,0x20,0x67,0xfc,0xde,
	0xa8,0xb0,0x9c,0x1a,0x93,0xd1,0x94,0xcd,0xbe,0x49,0x84,0x6e,0xb7,0x5d,0x5b,0x9a,
	0xd5,0x9a,0xec,0xb8,0x5b,0xf3,0xc9,0x17,0xfe,0xe9,0x42,0x48,0xde,0x8e,0xbe,0x96,
	0xb5,0xa9,0x32,0x8a,0x26,0x78,0xa6,0x47,0x98,0x31,0x22,0x29,0x2f,0x6c,0x79,0xb3,
	0x81,0x2c,0x81,0xad,0xda,0xdf,0x48,0xba,0x24,0x36,0x0a,0xf2,0xfa,0xb8,0xb4,0x64,
	0x98,0xc5,0xbf,0xc9,0xbe,0xbd,0x19,0x8e,0x26,0x8c,0x3b,0xa7,0x09,0xe0,0x42,0x14,
	0x68,0x00,0x7b,0xac,0xb2,0xdf,0x33,0x16,0x96,0xe9,0x39,0xe4,0x6c,0x51,0x8d,0x80,
	0xc8,0x14,0xe2,0x04,0x76,0xa9,0xfb,0x8a,0x50,0x25,0xc0,0x2d,0x59,0xc5,0x82,0x39,
	0xde,0x13,0x69,0x67,0x6c,0xcc,0x5a,0x71,0xfa,0x25,0x63,0x95,0x96,0x74,0xee,0x15,
	0x58,0x86,0xca,0x5d,0x2e,0x2f,0x31,0xd7,0x7e,0x0a,0xf1,0xfa,0x27,0xcf,0x73,0xc3,
	0x74,0x9c,0x47,0xab,0x18,0x50,0x1d,0xda,0xe2,0x75,0x7e,0x4f,0x74,0x01,0x90,0x5a,
	0xca,0xfa,0xaa,0xe3,0xe4,0xd5,0x9b,0x34,0x9a,0xdf,0x6a,0xce,0xbd,0x10,0x19,0x0d,
	0xfe,0x48,0x90,0xd1,0xe6,0x18,0x8d,0x0b,0x04,0x6d,0xf3,0x44,0x70,0x6c,0x63,0x1e
};

int length = 512;
int reps = 1 << 20;
int batch = 1;

template <typename block>
void print_transpose() {
	size_t n = sizeof(block);
	block* q = (block*) _mm_malloc(length * n, n);
	block* qT = (block*) _mm_malloc(length * n, n);
	PRG prg(fix_key);
	prg.random_block(q, length);
	for(size_t i = 0; i < n*8; i++) {
		for(size_t j = 0; j < length/(n*8); j++){
			print(q[i * (length/(n*8)) + j], "");
		}
		cout << endl;
	}
	cout << endl;
	for(size_t i = 0; i < n * 8 + 1; i++) {
		cout << "#";
	}
	cout << endl << endl;
	sse_trans((uint8_t *)(qT), (uint8_t*) q, n * 8, length);
	for(int i = 0; i < length; i++) {
		print(qT[i]);
	}
	_mm_free(q);
	_mm_free(qT);
}

double test_AES128(block* x) {
	AES_KEY aes;
	AES_set_encrypt_key(_mm_load_si128((__m128i*)fix_key), &aes);
	auto start = clock_start();
	for(int i = 0; i < reps/batch; i++){
		AES_ecb_encrypt_blks(x + i*batch, batch, &aes);
	}
	long long t = time_from(start);
	return t;
}
/*
double test_AES128_ks(block* x) {
	PRG prg;
	block* key = new block[1024];
	prg.random_block(key, 1024);
	ROUND_KEYS aes[batch];
	AES_KEY _aes;
	auto start = clock_start();
	switch (batch) {
		case 1:
			for(int i = 0; i < reps; i++){
				AES_set_encrypt_key(key[i % 1024], &_aes);
				AES_ecb_encrypt_blks(x + i, 1, &_aes);
			}
			break;
		case 2:
			for(int i = 0; i < reps/batch; i++){
				AES_ks2(key + (i*batch % 1024), aes);
				//AES_ecb_encrypt_blks(x + i*batch, batch, &aes);
				AES_ecb_ccr_ks2_enc2(x + i*batch, x + i*batch, aes);
			}
			break;
		case 4:
			break;
		case 8:
			for(int i = 0; i < reps/batch; i++){
				AES_ks8(key + (i*batch % 1024), aes);
				//AES_ecb_encrypt_blks(x + i*batch, batch, &aes);
				AES_ecb_ccr_ks8_enc8(x + i*batch, x + i*batch, aes);
			}
			break;
	}
	long long t = time_from(start);
	return t;
}
*/
/*
double test_AES256(block* x) {
	PRG prg;
	block256 key;
	prg.random_block(&key);
	AESNI_KEY aes;
	AESNI_set_encrypt_key(&aes, key);
	auto start = clock_start();
	for(int i = 0; i < reps/batch; i++){
		AESNI_ecb_encrypt_blks(x + i*batch, batch, &aes);
	}
	long long t = time_from(start);
	return t;
}

double test_AES256_ks(block* x) {
	PRG prg;
	block256* key = new block256[1024];
	prg.random_block(key, 1024);
	AESNI_KEY aes[batch];
	auto start = clock_start();
	for(int i = 0; i < reps/batch; i++){
		for(int j = 0; j < batch; j++){
			AESNI_set_encrypt_key(&aes[j], key[(i*batch + j) % 1024]);
		}
		AESNI_ecb_encrypt_blks_ks(x + i*batch, batch, aes);
	}
	long long t = time_from(start);
	delete[] key;
	return t;
}

double test_AES256_ks_new(block* x) {
	PRG prg;
	block256* key = new block256[1024];
	prg.random_block(key, 1024);
	//ROUND_KEYS aes[batch];
	AESNI_KEY aes[16];
	//AESNI_KEY _aes;
	auto start = clock_start();
	switch (batch) {
		case 1:
			for(int i = 0; i < reps; i++){
				//AES_set_encrypt_key(key[i % 1024], &_aes);
				//AES_ecb_encrypt_blks(x + i, 1, &_aes);
			}
			break;
		case 2:
			for(int i = 0; i < reps/batch; i++){
				AES_256_ks2(key + (i*batch % 1024), aes);
				AESNI_ecb_encrypt_blks_ks(x + i*batch, batch, aes);
				//AES_ecb_ccr_ks2_enc2(x + i*batch, x + i*batch, aes);
			}
			break;
		case 4:
			for(int i = 0; i < reps/batch; i++){
				for(int j = 0; j < batch/2; j++)
					AES_256_ks2(key + ((i*(batch)+2*j) % 1024), aes+2*j);
				AESNI_ecb_encrypt_blks_ks(x + i*batch, batch, aes);
				//AES_ecb_ccr_ks2_enc2(x + i*batch, x + i*batch, aes);
			}
			break;
		case 8:
			for(int i = 0; i < reps/batch; i++){
				AES_256_ks8(key + (i*batch % 1024), aes);
				AESNI_ecb_encrypt_blks_ks_x8(x + i*batch, batch, aes);
				//AES_ecb_ccr_ks2_enc2(x + i*batch, x + i*batch, aes);
			}
			break;
		case 16:
			for(int i = 0; i < reps/batch; i++){
				for(int j = 0; j < batch/8; j++)
					AES_256_ks8(key + ((i*(batch)+8*j) % 1024), aes+8*j);
				AESNI_ecb_encrypt_blks_ks(x + i*batch, batch, aes);
				//AES_ecb_ccr_ks2_enc2(x + i*batch, x + i*batch, aes);
			}
			break;
	}
	long long t = time_from(start);
	return t;
}

bool test_AES256_ks_new_correctness(){
	AESNI_KEY aes[8];
	AESNI_KEY aes_expected[8];
	block256* key = new block256[8];
	for(int i=0; i<8; i++){
		// set same input key for all 8 instances
		key[i] = _mm256_load_si256((const __m256i *)&AES256_TEST_KEY[0]);
	}
	for(int i=0; i<8; i++){
		aes_expected[i].rounds = 14;
		for(int j=0; j<15; j++){
			aes_expected[i].rk[j] = _mm_loadu_si128((const block *)&AES256_EXPECTED[16*j]);
		}
	}
	AES_256_ks8(key, aes);
	for(int j=0; j<8; j++){
		for(int i=0; i<15; i++){
			if((0 != memcmp(&(aes[j].rk[i]), &(aes_expected[j].rk[i]), sizeof(block)))){
				return false;
			}
		}
	}
	return true;
}*/

int main(int argc, char** argv) {
    int role;
    int bitlen;
    int num_ot;
    /*rgMapping amap;
    amap.arg("r", role);
    amap.arg("b", bitlen);
    amap.arg("N", num_ot);

    amap.parse(argc, argv);*/

    std::cout << "role: " << role << std::endl;
    std::cout << "bitlen: " << bitlen << std::endl;
    std::cout << "num_ot: " << num_ot << std::endl;

	PRG prg(fix_key);
	block* x = new block[reps];
	prg.random_block(x, reps);
	for(int i = 1; i <= 16; i*=2) {
		cout << "AES128 (Fixed Key) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES128(x)*1e6 << " Encps" << endl;
		batch *= 2;
	}
 /*	batch = 1;
	for(int i = 1; i <= 8; i*=2) {
		if (i != 4)
			cout << "AES128 (Variable Key) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES128_ks(x)*1e6 << " Encps" << endl;
		batch *= 2;
	}*/
	/*
	batch = 1;
	for(int i = 1; i <= 16; i*=2) {
		cout << "AES256 (Fixed Key) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES256(x)*1e6 << " Encps" << endl;
		batch *= 2;
	}
	batch = 1;
	for(int i = 1; i <= 16; i*=2) {
		cout << "AES256 (Variable Key Old) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES256_ks(x)*1e6 << " Encps" << endl;
		batch *= 2;
	}
	batch = 2;
	cout << "AES256 (Variable Key New) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES256_ks_new(x)*1e6 << " Encps" << endl;
	batch = 4;
	cout << "AES256 (Variable Key New) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES256_ks_new(x)*1e6 << " Encps" << endl;
	batch = 8;
	cout << "AES256 (Variable Key New) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES256_ks_new(x)*1e6 << " Encps" << endl;
	batch = 16;
	cout << "AES256 (Variable Key New) Encryption Time (batch=" << batch << "): " << double(reps)/test_AES256_ks_new(x)*1e6 << " Encps" << endl;
	assert(test_AES256_ks_new_correctness() == true);*/
	//cout << "Correctness of AES 256 Key Schedule is: CORRECT"<< endl;
	return 0;
}
