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

#ifndef UTILS_HE_H__
#define UTILS_HE_H__

#include "defines-HE.h"
#include "seal/seal.h"
#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"

// Taken from https://github.com/mc2-project/delphi/blob/master/rust/protocols-sys/c++/src/lib/conv2d.h
/* Helper function for performing modulo with possibly negative numbers */
inline int64_t neg_mod(int64_t val, int64_t mod) {
    return ((val % mod) + mod) % mod;
}

#define PRINT_NOISE_BUDGET(decryptor, ct, print_msg)                           \
  if (verbose)                                                                 \
  std::cout << "[Server] Noise Budget " << print_msg << ": " << YELLOW         \
            << decryptor->invariant_noise_budget(ct) << " bits" << RESET       \
            << std::endl

void generate_new_keys(int party, emp::NetIO *io, int slot_count,
                       std::shared_ptr<seal::SEALContext> &context_,
                       seal::Encryptor *&encryptor_,
                       seal::Decryptor *&decryptor_,
                       seal::Evaluator *&evaluator_,
                       seal::BatchEncoder *&encoder_,
                       seal::GaloisKeys *&gal_keys_,
                       seal::RelinKeys *& relin_keys_,
                       seal::Ciphertext *&zero_,
                       bool verbose = false);

void free_keys(int party, seal::Encryptor *&encryptor_,
               seal::Decryptor *&decryptor_, seal::Evaluator *&evaluator_,
               seal::BatchEncoder *&encoder_, seal::GaloisKeys *&gal_keys_,
               seal::RelinKeys *&relin_keys_,
               seal::Ciphertext *&zero_);

void send_encrypted_vector(emp::NetIO *io,
                           std::vector<seal::Ciphertext> &ct_vec);

void recv_encrypted_vector(emp::NetIO *io, std::shared_ptr<seal::SEALContext> &context_,
                           std::vector<seal::Ciphertext> &ct_vec);

void send_ciphertext(emp::NetIO *io, seal::Ciphertext &ct);

void recv_ciphertext(emp::NetIO *io, std::shared_ptr<seal::SEALContext> &context_, seal::Ciphertext &ct);

void set_poly_coeffs_uniform(
    uint64_t *poly, uint32_t bitlen,
    std::shared_ptr<seal::UniformRandomGenerator> random,
    std::shared_ptr<const seal::SEALContext::ContextData> &context_data);

void flood_ciphertext(
    seal::Ciphertext &ct,
    std::shared_ptr<const seal::SEALContext::ContextData> &context_data,
    uint32_t noise_len,
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

void random_mod_p(PRG &prg, uint64_t *arr, uint64_t size, uint64_t prime_mod);

uint64_t mod_mult(uint64_t a, uint64_t b, seal::Modulus mod);
#endif // UTILS_HE_H__
