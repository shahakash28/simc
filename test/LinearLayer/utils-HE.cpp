/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "utils-HE.h"
#include "seal/util/polyarithsmallmod.h"

using namespace std;
using namespace emp;
using namespace seal;
using namespace seal::util;


/* Helper function for performing modulo with possibly negative numbers */
/*inline int8_t neg_mod(int8_t val, int8_t mod) {
    return ((val % mod) + mod) % mod;
}*/

void generate_new_keys(int party, NetIO *io, int slot_count,
                       shared_ptr<SEALContext> &context_,
                       Encryptor *&encryptor_, Decryptor *&decryptor_,
                       Evaluator *&evaluator_, BatchEncoder *&encoder_,
                       GaloisKeys *&gal_keys_, RelinKeys *& relin_keys_, Ciphertext *&zero_,
                       bool verbose) {
  // auto context = SEALContext::Create(parms, true, sec_level_type::none);
  EncryptionParameters parms(scheme_type::bfv);
  parms.set_poly_modulus_degree(slot_count);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(slot_count));
  parms.set_plain_modulus(PLAINTEXT_MODULUS);

  context_ = shared_ptr<SEALContext>(new SEALContext(parms, false, sec_level_type::none));

  encoder_ = new BatchEncoder(*context_);
  evaluator_ = new Evaluator(*context_);
  if (party == BOB) {
    KeyGenerator keygen(*context_);
    PublicKey pub_key;
    keygen.create_public_key(pub_key);
    auto sec_key = keygen.secret_key();
    GaloisKeys gal_keys_;
    keygen.create_galois_keys(gal_keys_);
    RelinKeys relin_keys_;
    keygen.create_relin_keys(relin_keys_);

    stringstream os;
    pub_key.save(os);
    uint64_t pk_size = os.tellp();
    gal_keys_.save(os);
    uint64_t gk_size = (uint64_t)os.tellp() - pk_size;
    relin_keys_.save(os);
    uint64_t rk_size = (uint64_t)os.tellp() - (pk_size + gk_size);

    string keys_ser = os.str();
    io->send_data(&pk_size, sizeof(uint64_t));
    io->send_data(&gk_size, sizeof(uint64_t));
    io->send_data(&rk_size, sizeof(uint64_t));
    io->send_data(keys_ser.c_str(), pk_size + gk_size + rk_size);

#ifdef HE_DEBUG
    stringstream os_sk;
    sec_key.save(os_sk);
    uint64_t sk_size = os_sk.tellp();
    string keys_ser_sk = os_sk.str();
    io->send_data(&sk_size, sizeof(uint64_t));
    io->send_data(keys_ser_sk.c_str(), sk_size);
#endif
    encryptor_ = new Encryptor(*context_, pub_key);
    decryptor_ = new Decryptor(*context_, sec_key);
  } else // party == ALICE
  {
    uint64_t pk_size;
    uint64_t gk_size;
    uint64_t rk_size;
    io->recv_data(&pk_size, sizeof(uint64_t));
    io->recv_data(&gk_size, sizeof(uint64_t));
    io->recv_data(&rk_size, sizeof(uint64_t));

    char *key_share = new char[pk_size + gk_size + rk_size];
    io->recv_data(key_share, pk_size + gk_size + rk_size);
    stringstream is;
    PublicKey pub_key;
    is.write(key_share, pk_size);
    pub_key.load(*context_, is);
    gal_keys_ = new GaloisKeys();
    is.write(key_share + pk_size, gk_size);
    gal_keys_->load(*context_, is);
    relin_keys_ = new RelinKeys();
    is.write(key_share + pk_size + gk_size, rk_size);
    relin_keys_->load(*context_, is);
    delete[] key_share;

#ifdef HE_DEBUG
    uint64_t sk_size;
    io->recv_data(&sk_size, sizeof(uint64_t));
    char *key_share_sk = new char[sk_size];
    io->recv_data(key_share_sk, sk_size);
    stringstream is_sk;
    SecretKey sec_key;
    is_sk.write(key_share_sk, sk_size);
    sec_key.load(*context_, is_sk);
    delete[] key_share_sk;
    decryptor_ = new Decryptor(*context_, sec_key);
#endif
    encryptor_ = new Encryptor(*context_, pub_key);
    vector<uint64_t> pod_matrix(slot_count, 0ULL);
    Plaintext tmp;
    encoder_->encode(pod_matrix, tmp);
    zero_ = new Ciphertext;
    encryptor_->encrypt(tmp, *zero_);
  }
  if (verbose)
    cout << "Keys Generated (slot_count: " << slot_count << ")" << endl;
}

void free_keys(int party, Encryptor *&encryptor_, Decryptor *&decryptor_,
               Evaluator *&evaluator_, BatchEncoder *&encoder_,
               GaloisKeys *&gal_keys_, RelinKeys  *&relin_keys_, Ciphertext *&zero_) {
  delete encoder_;
  delete evaluator_;
  delete encryptor_;
  if (party == BOB) {
    delete decryptor_;
  } else // party ==ALICE
  {
#ifdef HE_DEBUG
    delete decryptor_;
#endif
    delete gal_keys_;
    delete relin_keys_;
    delete zero_;
  }
}

void send_encrypted_vector(NetIO *io, vector<Ciphertext> &ct_vec) {
  assert(ct_vec.size() > 0);
  stringstream os;
  uint64_t ct_size[ct_vec.size()];
  uint64_t prev_size=0;
  for (size_t ct = 0; ct < ct_vec.size(); ct++) {
    ct_vec[ct].save(os);
    ct_size[ct] = os.tellp() - prev_size;
    prev_size += ct_size[ct];
  }

  string ct_ser = os.str();
  for(int i=0; i<ct_vec.size(); i++) {
    io->send_data(&ct_size[i], sizeof(uint64_t));
  }
  io->send_data(ct_ser.c_str(), ct_ser.size());
}

void recv_encrypted_vector(NetIO *io, shared_ptr<SEALContext> &context_, vector<Ciphertext> &ct_vec) {
  assert(ct_vec.size() > 0);
  stringstream is;
  uint64_t ct_size[ct_vec.size()];
  uint64_t total_size=0;
  for(int i=0; i<ct_vec.size(); i++) {
    io->recv_data(&ct_size[i], sizeof(uint64_t));
    total_size += ct_size[i];
  }

  char *c_enc_result = new char[total_size];
  io->recv_data(c_enc_result, total_size);
  uint64_t prev_size=0;
  for (size_t ct = 0; ct < ct_vec.size(); ct++) {
    is.write(c_enc_result + prev_size, ct_size[ct]);
    prev_size += ct_size[ct];
    ct_vec[ct].unsafe_load(*context_, is);
  }
  delete[] c_enc_result;
}

void send_ciphertext(NetIO *io, Ciphertext &ct) {
  stringstream os;
  uint64_t ct_size;
  ct.save(os);
  ct_size = os.tellp();
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
}

void recv_ciphertext(NetIO *io, shared_ptr<SEALContext> &context_, Ciphertext &ct) {
  stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  ct.unsafe_load(*context_, is);
  delete[] c_enc_result;
}

void set_poly_coeffs_uniform(
    uint64_t *poly, uint32_t bitlen, shared_ptr<UniformRandomGenerator> random,
    shared_ptr<const SEALContext::ContextData> &context_data) {
  assert(bitlen < 128 && bitlen > 0);
  auto &parms = context_data->parms();
  auto &coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();
  uint64_t bitlen_mask = (1ULL << (bitlen % 64)) - 1;

  RandomToStandardAdapter engine(random);
  for (size_t i = 0; i < coeff_count; i++) {
    if (bitlen < 64) {
      uint64_t noise = (uint64_t(engine()) << 32) | engine();
      noise &= bitlen_mask;
      for (size_t j = 0; j < coeff_mod_count; j++) {
        poly[i + (j * coeff_count)] =
            barrett_reduce_64(noise, coeff_modulus[j]);
      }
    } else {
      uint64_t noise[2]; // LSB || MSB
      for (int j = 0; j < 2; j++) {
        noise[0] = (uint64_t(engine()) << 32) | engine();
        noise[1] = (uint64_t(engine()) << 32) | engine();
      }
      noise[1] &= bitlen_mask;
      for (size_t j = 0; j < coeff_mod_count; j++) {
        poly[i + (j * coeff_count)] =
            barrett_reduce_128(noise, coeff_modulus[j]);
      }
    }
  }
}

void flood_ciphertext(Ciphertext &ct,
                      shared_ptr<const SEALContext::ContextData> &context_data,
                      uint32_t noise_len, MemoryPoolHandle pool) {

  auto &parms = context_data->parms();
  auto &coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();

  auto noise(allocate_poly(coeff_count, coeff_mod_count, pool));
  shared_ptr<UniformRandomGenerator> random(parms.random_generator()->create());

  set_poly_coeffs_uniform(noise.get(), noise_len, random, context_data);
  for (size_t i = 0; i < coeff_mod_count; i++) {
    add_poly_coeffmod(noise.get() + (i * coeff_count),
                           ct.data() + (i * coeff_count), coeff_count,
                           coeff_modulus[i], ct.data() + (i * coeff_count));
  }

  set_poly_coeffs_uniform(noise.get(), noise_len, random, context_data);
  for (size_t i = 0; i < coeff_mod_count; i++) {
    add_poly_coeffmod(noise.get() + (i * coeff_count),
                           ct.data(1) + (i * coeff_count), coeff_count,
                           coeff_modulus[i], ct.data(1) + (i * coeff_count));
  }
}

void random_mod_p(PRG &prg, uint64_t *arr, uint64_t size, uint64_t prime_mod) {
      uint64_t boundary = (((-1 * prime_mod) / prime_mod) + 1) *
                   prime_mod; // prime_mod*floor((2^l)/prime_mod)
      int tries_before_resampling = 2;
      uint64_t size_total = tries_before_resampling * size;
      uint64_t *randomness = new uint64_t[size_total];
      uint64_t rptr = 0, arrptr = 0;
      while (arrptr < size) {
        prg.random_data(randomness, sizeof(uint64_t) * size_total);
        rptr = 0;
        for (; (arrptr < size) && (rptr < size_total); arrptr++, rptr++) {
          while (randomness[rptr] > boundary) {
            rptr++;
            if (rptr >= size_total) {
              prg.random_data(randomness, sizeof(uint64_t) * size_total);
              rptr = 0;
            }
          }
          arr[arrptr] = randomness[rptr] % prime_mod;
        }
      }
      delete[] randomness;
}

uint64_t mod_mult(uint64_t a, uint64_t b, seal::Modulus mod) {
  unsigned long long temp_result[2];
  seal::util::multiply_uint64(a, b, temp_result);
  uint64_t result = seal::util::barrett_reduce_128(temp_result, mod);
  return result;
}
