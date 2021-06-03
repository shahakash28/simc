#pragma once

#include "LinearLayer/fc-field.h"
#include "LinearLayer/defines-HE.h"

using namespace std;
using namespace emp;
using namespace seal;

enum neural_net {
  NONE,
  MINIONN,
  CIFAR10
};
neural_net choice_nn;
neural_net def_nn = NONE;

long long total_time = 0;

int32_t bitlength = 44;
uint64_t prime_mod = PLAINTEXT_MODULUS;
int party = 0;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int num_rows = 512;
int common_dim = 1024;
int filter_precision = 15;

seal::Modulus mod(prime_mod);

void LinearLayerFirstFC(FCField &he_fc, int32_t num_rows, int32_t common_dim) {
  int num_cols = 1;

  //Setup Input objects
  vector<vector<uint64_t>> inputs;

  //Setup Output shares objects
  vector<vector<uint64_t>> op_shares(num_rows);
  vector<vector<uint64_t>> mac_op_shares(num_rows);
  for (int i = 0; i < num_rows; i++) {
    op_shares[i].resize(num_cols);
    mac_op_shares[i].resize(num_cols);
  }
  PRG prg;

  //Prepare Dummy Inputs
  if(party == ALICE) {
    inputs.resize(num_rows, vector<uint64_t>(common_dim, 0));
    //Create input matrix
    for(int i=0; i< num_rows; i++) {
      random_mod_p(prg, inputs[i].data(), common_dim, prime_mod);
    }
  } else {
    //Create input vector
    inputs.resize(common_dim, vector<uint64_t>(num_cols, 0));
    for(int i=0; i<common_dim; i++) {
      random_mod_p(prg, inputs[i].data(), num_cols, prime_mod);
    }
  }
  he_fc.matrix_multiplication_first(num_rows, common_dim, num_cols, inputs, op_shares, mac_op_shares, mod, false,
                              false);
}

void LinearLayerFC(FCField &he_fc, int32_t num_rows, int32_t common_dim) {
  int num_cols = 1;

  //Setup Input objects
  vector<vector<uint64_t>> matrix;
  vector<vector<uint64_t>> input_share;
  vector<vector<uint64_t>> mac_input_share;

  PRG prg;
  //Prepare Dummy Inputs
  input_share.resize(common_dim, vector<uint64_t>(num_cols, 0));
  for(int i=0; i<common_dim; i++) {
    random_mod_p(prg, input_share[i].data(), num_cols, prime_mod);
  }

  mac_input_share.resize(common_dim, vector<uint64_t>(num_cols, 0));
  for(int i=0; i<common_dim; i++) {
    random_mod_p(prg, mac_input_share[i].data(), num_cols, prime_mod);
  }

  if(party == ALICE) {
    matrix.resize(num_rows, vector<uint64_t>(common_dim, 0));
    //Create input matrix
    for(int i=0; i< num_rows; i++) {
      random_mod_p(prg, matrix[i].data(), common_dim, prime_mod);
    }
  }
  auto start = clock_start();
  he_fc.matrix_multiplication_gen(num_rows, common_dim, num_cols, matrix, input_share, mac_input_share, mod, false,
                              false);
  long long t = time_from(start);
  total_time += t;
}

void parse_arguments(int argc, char**arg, int *party, int *port) {
  *party = atoi (arg[1]);
   address = arg[2];
	*port = atoi (arg[3]);
  if(argc < 5) {
    choice_nn = def_nn;
  } else {
    choice_nn = neural_net(atoi (arg[4]));
  }
}

int main(int argc, char** argv){
  parse_arguments(argc, argv, &party, &port);

  NetIO * io = new NetIO(party==ALICE ? nullptr : address.c_str(), port);

  int slot_count = POLY_MOD_DEGREE;
  shared_ptr<SEALContext> context;
  Encryptor *encryptor;
  Decryptor *decryptor;
  Evaluator *evaluator;
  BatchEncoder *encoder;
  GaloisKeys *galois_keys;
  Ciphertext *zero;

  //Generate Keys
  //generate_new_keys(party, io, slot_count, context, encryptor, decryptor, evaluator, encoder, galois_keys, zero);
  /*if(party == ALICE) {
    //Generate Server's random input

  } else {
    //Generate Client's random input
    for (int i = 0; i < common_dim; i++) {
      B[i].resize(1);
      random_mod_p(prg, B[i].data(), num_cols, prime_mod);
    }
  }*/
  uint64_t comm_sent = 0;
  uint64_t start_comm = io->counter;
  auto start = clock_start();
  start_comm = io->counter;
  FCField he_fc(party, io);
  long long t = time_from(start);
  total_time += t;
  if(choice_nn==MINIONN) {
    num_rows = 100;
    common_dim = 256;
    LinearLayerFC(he_fc, num_rows, common_dim);
    num_rows = 10;
    common_dim = 100;
    LinearLayerFC(he_fc, num_rows, common_dim);
  } else {
    num_rows = 10;
    common_dim = 1024;
    LinearLayerFC(he_fc, num_rows, common_dim);
  }
  cout << "######################Performance#######################" <<endl;
  cout<<"Time Taken: "<<total_time<<" mus"<<endl;
  //Calculate Communication
  comm_sent = (io->counter-start_comm)>>20;
  cout<<"Sent Data (MB): "<<comm_sent<<endl;
  cout << "########################################################" <<endl;
  //LinearLayerFirstFC(he_fc, num_rows, common_dim);
}
