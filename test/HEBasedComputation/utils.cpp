
#include "utils.h"


void key_generator(int party, NetIO* io, SEALContext context, Encryptor *&encryptor, Decryptor *&decryptor, Evaluator *&evaluator, BatchEncoder *&encoder, GaloisKeys *&galois_keys, Ciphertext *&zero) {
    encoder = new BatchEncoder(context);
    evaluator = new Evaluator(context);

    if(party == BOB) {
      //Client generates HE Keys
      KeyGenerator keygen(context);
      SecretKey secret_key = keygen.secret_key();
      PublicKey public_key;
      keygen.create_public_key(public_key);

      GaloisKeys galois_keys;
      keygen.create_galois_keys(galois_keys);

      stringstream os;
      public_key.save(os);
      uint64_t pk_size = os.tellp();
      galois_keys.save(os);
      uint64_t gk_size = (uint64_t)os.tellp() - pk_size;

       //Send keys to server
      string keys_ser = os.str();
      io->send_data(&pk_size, sizeof(uint64_t));
      io->send_data(&gk_size, sizeof(uint64_t));
      io->send_data(keys_ser.c_str(), pk_size + gk_size);

     encryptor = new Encryptor(context, public_key);
     decryptor = new Decryptor(context, secret_key);
    } else {
      //Receive keys from client
     uint64_t pk_size;
     uint64_t gk_size;
     io->recv_data(&pk_size, sizeof(uint64_t));
     io->recv_data(&gk_size, sizeof(uint64_t));
     char *key_share = new char[pk_size + gk_size];
     io->recv_data(key_share, pk_size + gk_size);
     //Load keys from received data
     stringstream is;
     PublicKey public_key;
     is.write(key_share, pk_size);
     public_key.load(context, is);
     galois_keys = new GaloisKeys();
     is.write(key_share + pk_size, gk_size);
     galois_keys->load(context, is);
     delete[] key_share;

     encryptor = new Encryptor(context, public_key);
     vector<uint64_t> pod_matrix(POLY_MOD_DEGREE, 0ULL);

     Plaintext tmp;
     encoder->encode(pod_matrix, tmp);
     zero = new Ciphertext;
     encryptor->encrypt(tmp, *zero);
    }
}

void free_keys(int party, Encryptor *&encryptor, Decryptor *&decryptor, Evaluator *&evaluator, BatchEncoder *&encoder, GaloisKeys *&gal_keys, Ciphertext *&zero) {
  delete encoder;
  delete evaluator;
  delete encryptor;
  if (party == BOB) {
    delete decryptor;
  }
  else // party ==ALICE
  {
    delete gal_keys;
    delete zero;
  }
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

void recv_ciphertext(NetIO *io, SEALContext context, Ciphertext &ct) {
  stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  ct.unsafe_load(context, is);
  delete[] c_enc_result;
}
