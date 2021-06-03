
#include "emp-sh2pc/emp-sh2pc.h"
#include "seal/seal.h"

using namespace std;
using namespace seal;
using namespace emp;

//44 bit prime
const uint64_t PLAINTEXT_MODULUS = 17592060215297;
const uint64_t POLY_MOD_DEGREE = 8192;

void key_generator(int party, NetIO* io, SEALContext context, Encryptor *&encryptor, Decryptor *&decryptor, Evaluator *&evaluator, BatchEncoder *&encoder, GaloisKeys *&galois_keys, RelinKeys *& relin_keys, Ciphertext *&zero);
void free_keys(int party, seal::Encryptor *&encryptor, seal::Decryptor *&decryptor, seal::Evaluator *&evaluator, seal::BatchEncoder *&encoder, seal::GaloisKeys *&gal_keys, seal::Ciphertext *&zero);

void send_ciphertext(NetIO *io, Ciphertext &ct);
void recv_ciphertext(NetIO *io, SEALContext context, Ciphertext &ct);
