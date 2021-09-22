#define MAX_THREADS 8

//Slackoverflow Code For bit-wise shift
#define SHL128(v, n) \
({ \
    __m128i v1, v2; \
 \
    if ((n) >= 64) \
    { \
        v1 = _mm_slli_si128(v, 8); \
        v1 = _mm_slli_epi64(v1, (n) - 64); \
    } \
    else \
    { \
        v1 = _mm_slli_epi64(v, n); \
        v2 = _mm_slli_si128(v, 8); \
        v2 = _mm_srli_epi64(v2, 64 - (n)); \
        v1 = _mm_or_si128(v1, v2); \
    } \
    v1; \
})

uint64_t mod_shift(uint64_t a, uint64_t b, uint64_t prime_mod) {
    __m128i temp, stemp;
    memcpy(&temp, &a, 8);
    stemp = SHL128(temp, b);

    uint64_t input[2];
    input[0] = stemp[0];
    input[1] = stemp[1];

    uint64_t result = seal::util::barrett_reduce_128(input, mod);

    return result;
}

uint64_t mod_mult(uint64_t a, uint64_t b) {
  unsigned long long temp_result[2];
  seal::util::multiply_uint64(a, b, temp_result);

  /*uint64_t input[2];
  input[0] = res[0];
  input[1] = res[1];*/
  uint64_t result = seal::util::barrett_reduce_128(temp_result, mod);
  return result;
}

//Referred SCI OT repo's logic to pack ot messages
void pack_decryption_table(uint64_t *pack_table, uint64_t *ciphertexts, int pack_size, int batch_size, int bitlen) {
  uint64_t beg_idx = 0;
  uint64_t end_idx = 0;
  uint64_t beg_blk = 0;
  uint64_t end_blk = 0;
  uint64_t temp_blk = 0;
  uint64_t mask = (1ULL << bitlen) - 1;
  uint64_t pack_blk_size = 64;

  if (bitlen == 64)
    mask = -1;

  for (int i = 0; i < pack_size; i++) {
    pack_table[i] = 0;
  }

  for (int i = 0; i < batch_size; i++) {
    beg_idx = i * bitlen;
    end_idx = beg_idx + bitlen;
    end_idx -= 1;
    beg_blk = beg_idx / pack_blk_size;
    end_blk = end_idx / pack_blk_size;

    if (beg_blk == end_blk) {
      pack_table[beg_blk] ^= (ciphertexts[i] & mask) << (beg_idx % pack_blk_size);
    } else {
      temp_blk = (ciphertexts[i] & mask);
      pack_table[beg_blk] ^= (temp_blk) << (beg_idx % pack_blk_size);
      pack_table[end_blk] ^= (temp_blk) >> (pack_blk_size - (beg_idx % pack_blk_size));
    }
  }
}

//Referred SCI OT repo's logic to unpack ot messages
void unpack_decryption_table(uint64_t *pack_table, uint64_t *ciphertexts, int pack_size, int batch_size, int bitlen) {
  uint64_t beg_idx = 0;
  uint64_t end_idx = 0;
  uint64_t beg_blk = 0;
  uint64_t end_blk = 0;
  uint64_t temp_blk = 0;
  uint64_t mask = (1ULL << bitlen) - 1;
  uint64_t pack_blk_size = 64;

  for (int i = 0; i < batch_size; i++) {
    beg_idx = i * bitlen;
    end_idx = beg_idx + bitlen - 1;
    beg_blk = beg_idx / pack_blk_size;
    end_blk = end_idx / pack_blk_size;

    if (beg_blk == end_blk) {
      ciphertexts[i] = (pack_table[beg_blk] >> (beg_idx % pack_blk_size)) & mask;
    } else {
      ciphertexts[i] = 0;
      ciphertexts[i] ^= (pack_table[beg_blk] >> (beg_idx % pack_blk_size));
      ciphertexts[i] ^= (pack_table[end_blk] << (pack_blk_size - (beg_idx % pack_blk_size)));
      ciphertexts[i] = ciphertexts[i] & mask;
    }
  }
}

void create_ciphertexts(Integer *garbled_data, block label_delta, uint64_t *ciphertexts, uint64_t* server_shares, int bitlen, int nrelu, uint64_t alpha, int l_idx) {
  uint64_t delta_int;
  memcpy(&delta_int, &label_delta[l_idx], 8);

  uint64_t mask = (1ULL << bitlen) - 1;
  uint64_t label_temp;
  uint64_t **random_val = (uint64_t **)malloc(nrelu*sizeof(uint64_t*));
  uint8_t pnp, cpnp;
  for(int i=0; i<nrelu; i++) {
    random_val[i] = (uint64_t *)malloc(bitlen*sizeof(uint64_t));
  }

  for(int i=0; i<nrelu; i++) {
    server_shares[i] = 0;
    for(int j=0; j<bitlen; j++) {
      memcpy(&label_temp, &garbled_data[i].bits[j].bit[l_idx], 8);
      prg.random_data(&random_val[i][j], 8);
      random_val[i][j] %= prime_mod;
      pnp = (garbled_data[i].bits[j].bit[0]) & 1;
      cpnp = 1 - pnp;
      ciphertexts[(i*bitlen+j)*2+pnp] = (random_val[i][j])^(label_temp & mask);
      ciphertexts[(i*bitlen+j)*2+cpnp] = ((random_val[i][j]+alpha)%prime_mod)^((label_temp^delta_int) & mask);
      if(i==0) {
        uint64_t l1 = label_temp & mask, l2 = label_temp^delta_int & mask;
      }
      server_shares[i] = (server_shares[i] + mod_shift(random_val[i][j],j,prime_mod))%prime_mod;
    }
    server_shares[i] = prime_mod - server_shares[i];
  }
}

void decrypt_ciphertexts(Integer *garbled_data, uint64_t *ciphertexts, uint64_t* client_shares, int bitlen, int nrelu, int l_idx) {
  uint64_t label_temp;
  uint8_t pnp;
  uint64_t random_val;

  uint64_t mask = (1ULL << bitlen) - 1;

  for(int i=0; i<nrelu; i++) {
    client_shares[i] = 0;
    for(int j=0; j< bitlen; j++) {
      memcpy(&label_temp, &garbled_data[i].bits[j].bit[l_idx], 8);
      pnp = (garbled_data[i].bits[j].bit[0]) & 1;
      random_val = ciphertexts[(i*bitlen+j)*2+pnp]^(label_temp & mask);
      client_shares[i] = (client_shares[i] + mod_shift(random_val,j,prime_mod))%prime_mod;
    }
  }
}
