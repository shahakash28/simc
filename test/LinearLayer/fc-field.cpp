/*
Original Author: ryanleh
Modified Work Copyright (c) 2020 Microsoft Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Modified by Deevashwer Rathee
*/

#include "fc-field.h"

using namespace std;
using namespace seal;
using namespace emp;

Ciphertext preprocess_vec(const uint64_t *input, const FCMetadata &data,
                          Encryptor &encryptor, BatchEncoder &batch_encoder) {
  // Create copies of the input vector to fill the ciphertext appropiately.
  // Pack using powers of two for easy rotations later
  vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
  uint64_t size_pow2 = next_pow2(data.image_size);
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int col = 0; col < data.image_size; col++) {
    for (int idx = 0; idx < data.pack_num; idx++) {
      pod_matrix[col + size_pow2 * idx] = input[col];
    }
  }

  Ciphertext ciphertext;
  Plaintext tmp;
  batch_encoder.encode(pod_matrix, tmp);
  encryptor.encrypt(tmp, ciphertext);
  return ciphertext;
}

Plaintext preprocess_vec_plain(const uint64_t *input, const FCMetadata &data,
                          BatchEncoder &batch_encoder) {
  // Create copies of the input vector to fill the ciphertext appropiately.
  // Pack using powers of two for easy rotations later
  vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
  uint64_t size_pow2 = next_pow2(data.image_size);
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int col = 0; col < data.image_size; col++) {
    for (int idx = 0; idx < data.pack_num; idx++) {
      pod_matrix[col + size_pow2 * idx] = input[col];
    }
  }

  Plaintext plaintext;
  batch_encoder.encode(pod_matrix, plaintext);
  return plaintext;
}

vector<Plaintext> preprocess_matrix(const uint64_t *const *matrix,
                                    const FCMetadata &data,
                                    BatchEncoder &batch_encoder) {
  // Pack the filter in alternating order of needed ciphertexts. This way we
  // rotate the input once per ciphertext
  vector<vector<uint64_t>> mat_pack(data.inp_ct,
                                    vector<uint64_t>(data.slot_count, 0ULL));
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int row = 0; row < data.filter_h; row++) {
    int ct_idx = row / data.inp_ct;
    for (int col = 0; col < data.filter_w; col++) {
      mat_pack[row % data.inp_ct][col + next_pow2(data.filter_w) * ct_idx] =
          matrix[row][col];
    }
  }

  // Take the packed ciphertexts above and repack them in a diagonal ordering.
  int mod_mask = (data.inp_ct - 1);
  int wrap_thresh = min(data.slot_count >> 1, next_pow2(data.filter_w));
  int wrap_mask = wrap_thresh - 1;
  vector<vector<uint64_t>> mat_diag(data.inp_ct,
                                    vector<uint64_t>(data.slot_count, 0ULL));
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct = 0; ct < data.inp_ct; ct++) {
    for (int col = 0; col < data.slot_count; col++) {
      int ct_diag_l = (col - ct) & wrap_mask & mod_mask;
      int ct_diag_h = (col ^ ct) & (data.slot_count / 2) & mod_mask;
      int ct_diag = (ct_diag_h + ct_diag_l);

      int col_diag_l = (col - ct_diag_l) & wrap_mask;
      int col_diag_h = wrap_thresh * (col / wrap_thresh) ^ ct_diag_h;
      int col_diag = col_diag_h + col_diag_l;

      mat_diag[ct_diag][col_diag] = mat_pack[ct][col];
    }
  }

  vector<Plaintext> enc_mat(data.inp_ct);
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct = 0; ct < data.inp_ct; ct++) {
    batch_encoder.encode(mat_diag[ct], enc_mat[ct]);
  }
  return enc_mat;
}

/* Generates a masking vector of random noise that will be applied to parts of
 * the ciphertext that contain leakage */
Plaintext fc_preprocess_noise(const uint64_t *secret_share,
                               const FCMetadata &data,
                               BatchEncoder &batch_encoder) {
  // Sample randomness into vector
  vector<uint64_t> noise(data.slot_count, 0ULL);
  PRG prg;
  random_mod_p(prg, noise.data(), data.slot_count, prime_mod);

  // Puncture the vector with secret shares where an actual fc result value
  // lives
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int row = 0; row < data.filter_h; row++) {
    int curr_set = row / data.inp_ct;
    noise[(row % data.inp_ct) + next_pow2(data.image_size) * curr_set] =
        secret_share[row];
  }

  Plaintext enc_noise;
  batch_encoder.encode(noise, enc_noise);

  return enc_noise;
}

Ciphertext fc_online(Ciphertext &ct, vector<Plaintext> &enc_mat,
                     const FCMetadata &data, Evaluator &evaluator,
                     GaloisKeys &gal_keys, RelinKeys &relin_keys, Ciphertext &zero) {
  Ciphertext result = zero;
  // For each matrix ciphertext, rotate the input vector once and multiply + add
  Ciphertext tmp;
  for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
      evaluator.rotate_rows(ct, ct_idx, gal_keys, tmp);
      evaluator.multiply_plain_inplace(tmp, enc_mat[ct_idx]);
      evaluator.relinearize_inplace(tmp, relin_keys);
      evaluator.add_inplace(result, tmp);
  }

  // Rotate all partial sums together
  for (int rot = data.inp_ct; rot < next_pow2(data.image_size); rot *= 2) {
    Ciphertext tmp;
    if (rot == data.slot_count / 2) {
      evaluator.rotate_columns(result, gal_keys, tmp);
    } else {
      evaluator.rotate_rows(result, rot, gal_keys, tmp);
    }
    evaluator.add_inplace(result, tmp);
  }

  return result;
}

uint64_t *fc_postprocess(Ciphertext &ct, const FCMetadata &data,
                         BatchEncoder &batch_encoder, Decryptor &decryptor) {
  vector<uint64_t> plain(data.slot_count, 0ULL);
  Plaintext tmp;
  decryptor.decrypt(ct, tmp);
  batch_encoder.decode(tmp, plain);

  uint64_t *result = new uint64_t[data.filter_h];
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int row = 0; row < data.filter_h; row++) {
    int curr_set = row / data.inp_ct;
    result[row] =
        plain[(row % data.inp_ct) + next_pow2(data.image_size) * curr_set];
  }
  return result;
}

uint64_t *fc_postprocess_mac(Ciphertext &ct, const FCMetadata &data,
                         BatchEncoder &batch_encoder, Decryptor &decryptor) {
  vector<uint64_t> plain;
  Plaintext tmp;
  decryptor.decrypt(ct, tmp);
  batch_encoder.decode(tmp, plain);

  uint64_t *result = new uint64_t[data.image_size];
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int col = 0; col < data.image_size; col++) {
       result[col] = plain[col];
  }
  return result;
}

FCField::FCField(int party, NetIO *io) {
  this->party = party;
  this->io = io;
  this->slot_count = POLY_MOD_DEGREE;
  generate_new_keys(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero);
}

FCField::~FCField() {
  free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, relin_keys, zero);
}

void FCField::configure() {
  data.slot_count = slot_count;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
  data.inp_ct_1 = ceil((float)next_pow2(data.filter_w) / data.pack_num);
}

vector<uint64_t> FCField::ideal_functionality(uint64_t *vec,
                                              uint64_t **matrix, seal::Modulus mod) {
  vector<uint64_t> result(data.filter_h, 0ULL);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      uint64_t partial = mod_mult(vec[idx], matrix[row][idx], mod);
      result[row] = (result[row] + partial)%prime_mod;
    }
  }
  return result;
}

void FCField::matrix_multiplication_first(int32_t num_rows, int32_t common_dim,
                                    int32_t num_cols,
                                    vector<vector<uint64_t>> &inputs,
                                    vector<vector<uint64_t>> &op_shares,
                                    vector<vector<uint64_t>> &mac_op_shares,
                                    seal::Modulus mod,
                                    bool verify_output, bool verbose) {

  assert(num_cols == 1);
  data.filter_h = num_rows;
  data.filter_w = common_dim;
  data.image_size = common_dim;
  this->slot_count =
      min(max(8192, 2 * next_pow2(common_dim)), SEAL_POLY_MOD_DEGREE_MAX);
  configure();

  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  RelinKeys *relin_keys_;
  Ciphertext *zero_;
  if (slot_count > POLY_MOD_DEGREE) {
    generate_new_keys(party, io, slot_count, context, encryptor_, decryptor_,
                      evaluator_, encoder_, gal_keys_, relin_keys_, zero_);
  } else {
    encryptor_ = this->encryptor;
    decryptor_ = this->decryptor;
    evaluator_ = this->evaluator;
    encoder_ = this->encoder;
    gal_keys_ = this->gal_keys;
    relin_keys_ = this->relin_keys;
    zero_ = this->zero;
  }

  if(party == BOB) {
    vector<uint64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = inputs[i][0];
    }

    if (verbose)
      cout << "[Client] Vector Generated" << endl;

    auto ct = preprocess_vec(vec.data(), data, *encryptor_, *encoder_);
    send_ciphertext(io, ct);
    if (verbose)
      cout << "[Client] Vector processed and sent" << endl;

    Ciphertext linear;
    Ciphertext linear_mac;
    recv_ciphertext(io, context, linear);
    recv_ciphertext(io, context, linear_mac);
    if (verbose)
      cout << "[Client] Receive ciphertexts of shares" << endl;

    auto linear_1 = fc_postprocess(linear, data, *encoder_, *decryptor_);
    auto linear_mac_1 = fc_postprocess(linear_mac, data, *encoder_, *decryptor_);

    if (verbose)
      cout << "[Client] Obtain shares" << endl;

    if(verify_output) {
      io->send_data(vec.data(), sizeof(uint64_t) * common_dim);
      io->send_data(linear_1, sizeof(uint64_t) * num_rows);
      io->send_data(linear_mac_1, sizeof(uint64_t) * num_rows);
    }
  } else {
    PRG prg;
    //Generate Random MAC
    uint64_t mac_key;
    random_mod_p(prg, &mac_key, 1, prime_mod);
    vector<uint64_t> mac_vec(encoder_->slot_count(), mac_key);
    Plaintext* enc_mac = new Plaintext();
    encoder_->encode(mac_vec, *enc_mac);

    //Generate random_shares
    vector<uint64_t> op_shares_vec(num_rows, 0);
    vector<uint64_t> mac_op_shares_vec(num_rows, 0);
    random_mod_p(prg, op_shares_vec.data(), num_rows, prime_mod);
    random_mod_p(prg, mac_op_shares_vec.data(), num_rows, prime_mod);

    Plaintext linear_0 = fc_preprocess_noise(op_shares_vec.data(), data, *encoder_);
    Plaintext linear_mac_0 = fc_preprocess_noise(mac_op_shares_vec.data(), data, *encoder_);
    if (verbose)
      cout << "[Server] Generate shares" << endl;

    //Preprocess Matrix
    vector<uint64_t *> matrix(num_rows);
    for (int i = 0; i < num_rows; i++) {
      matrix[i] = new uint64_t[common_dim];
      for (int j = 0; j < common_dim; j++) {
        matrix[i][j] = inputs[i][j];
      }
    }
    auto encoded_mat = preprocess_matrix(matrix.data(), data, *encoder_);
    if (verbose)
      cout << "[Server] Preprocess Matrix" << endl;


    //Receive ciphertext from client
    Ciphertext ct;
    recv_ciphertext(io, context, ct);
    if (verbose)
      cout << "[Server] Receive Ciphertext from client" << endl;

    //Compute FC component
    Ciphertext linear = fc_online(ct, encoded_mat, data, *evaluator_, *gal_keys_, *relin_keys_,
                               *zero_);
    if (verbose)
      cout << "[Server] Compute FC" << endl;

    Ciphertext linear_mac;
    evaluator_->multiply_plain(linear, *enc_mac, linear_mac);
    //Linear Share
    evaluator_->sub_plain_inplace(linear, linear_0);
    //Linear MAC Share
    evaluator_->sub_plain_inplace(linear_mac, linear_mac_0);
    if (verbose)
      cout << "[Server] Generate Client shares" << endl;

    //Send ciphertexts to client
    send_ciphertext(io, linear);
    send_ciphertext(io, linear_mac);
    if (verbose)
      cout << "[Server] Send ciphertexts of shares to client" << endl;

    //Verify
    if(verify_output) {
      vector<uint64_t>  vec(common_dim);
      //receive client input
      io->recv_data(vec.data(), sizeof(uint64_t)*common_dim);
      //Compute FC
      auto result_actual = ideal_functionality(vec.data(), matrix.data(), mod);
      //Compute MAC
      vector<uint64_t> result_actual_mac(num_rows);

      for(int i=0;i<num_rows; i++) {
        result_actual_mac[i] = mod_mult(mac_key,result_actual[i], mod);
      }

      //receive client_shares
      vector<uint64_t> op_shares_vec_1(num_rows);
      vector<uint64_t> mac_op_shares_vec_1(num_rows);
      io->recv_data(op_shares_vec_1.data(), sizeof(uint64_t)*num_rows);
      io->recv_data(mac_op_shares_vec_1.data(), sizeof(uint64_t)*num_rows);
      // reconstruct output
      for(int i=0; i<num_rows; i++) {
        op_shares_vec_1[i] = (op_shares_vec[i] + op_shares_vec_1[i])%prime_mod;
        mac_op_shares_vec_1[i] = (mac_op_shares_vec[i] + mac_op_shares_vec_1[i])%prime_mod;
      }

      //check for equality
      int ctr=0;
      int mctr=0;
      for(int i=0;i<num_rows; i++) {
        if(result_actual[i] == op_shares_vec_1[i])
          ctr++;
        if(result_actual_mac[i] == mac_op_shares_vec_1[i])
          mctr++;
      }
      cout<<"Correct Shares: "<< ctr<<endl;
      cout<<"Correct Mac Shares: "<<mctr<<endl;
    }
  }
}

void FCField::matrix_multiplication_gen(int32_t num_rows, int32_t common_dim,
                                    int32_t num_cols,
                                    vector<vector<uint64_t>> &matrix,
                                    vector<vector<uint64_t>> &input_share,
                                    vector<vector<uint64_t>> &mac_input_share,
                                    seal::Modulus mod,
                                    bool verify_output, bool verbose) {

  assert(num_cols == 1);
  data.filter_h = num_rows;
  data.filter_w = common_dim;
  data.image_size = common_dim;
  this->slot_count =
      min(max(8192, 2 * next_pow2(common_dim)), SEAL_POLY_MOD_DEGREE_MAX);
  configure();

  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  RelinKeys *relin_keys_;
  Ciphertext *zero_;
  if (slot_count > POLY_MOD_DEGREE) {
    generate_new_keys(party, io, slot_count, context, encryptor_, decryptor_,
                      evaluator_, encoder_, gal_keys_, relin_keys_, zero_);
  } else {
    encryptor_ = this->encryptor;
    decryptor_ = this->decryptor;
    evaluator_ = this->evaluator;
    encoder_ = this->encoder;
    gal_keys_ = this->gal_keys;
    relin_keys_ = this->relin_keys;
    zero_ = this->zero;
  }

  if(party == BOB) {
    vector<uint64_t> input_vec(common_dim);
    vector<uint64_t> mac_input_vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      input_vec[i] = input_share[i][0];
      mac_input_vec[i] = mac_input_share[i][0];
    }
    if (verbose)
      cout << "[Client] Vector Generated" << endl;

    auto input_ct = preprocess_vec(input_vec.data(), data, *encryptor_, *encoder_);
    auto mac_input_ct = preprocess_vec(mac_input_vec.data(), data, *encryptor_, *encoder_);
    send_ciphertext(io, input_ct);
    send_ciphertext(io, mac_input_ct);

    if (verbose)
      cout << "[Client] Vector processed and sent" << endl;

    Ciphertext linear;
    Ciphertext linear_mac;
    Ciphertext mac_ver_ip;
    recv_ciphertext(io, context, linear);
    recv_ciphertext(io, context, linear_mac);
    recv_ciphertext(io, context, mac_ver_ip);
    if (verbose)
      cout << "[Client] Receive ciphertexts of shares" << endl;

    auto linear_1 = fc_postprocess(linear, data, *encoder_, *decryptor_);
    auto linear_mac_1 = fc_postprocess(linear_mac, data, *encoder_, *decryptor_);
    auto mac_ver_ip_1 = fc_postprocess_mac(mac_ver_ip, data, *encoder_, *decryptor_);

    if (verbose)
      cout << "[Client] Obtain shares" << endl;

    if(verify_output) {
      io->send_data(input_vec.data(), sizeof(uint64_t) * common_dim);
      io->send_data(mac_input_vec.data(), sizeof(uint64_t) * common_dim);
      io->send_data(linear_1, sizeof(uint64_t) * num_rows);
      io->send_data(linear_mac_1, sizeof(uint64_t) * num_rows);
      io->send_data(mac_ver_ip_1, sizeof(uint64_t) * common_dim);
    }
  } else {
    vector<uint64_t> input_vec(common_dim);
    vector<uint64_t> mac_input_vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      input_vec[i] = input_share[i][0];
      mac_input_vec[i] = mac_input_share[i][0];
    }

    Plaintext input_pt = preprocess_vec_plain(input_vec.data(), data, *encoder_);
    Plaintext mac_input_pt = preprocess_vec_plain(mac_input_vec.data(), data, *encoder_);

    PRG prg;
    //Generate Random MAC
    uint64_t mac_key;
    random_mod_p(prg, &mac_key, 1, prime_mod);
    uint64_t mac_key_sqrt = mod_mult(mac_key, mac_key, mod);
    uint64_t mac_key_cube = mod_mult(mac_key_sqrt, mac_key, mod);

    vector<uint64_t> mac_vec_sqrt(encoder_->slot_count(), mac_key_sqrt);
    Plaintext* enc_mac_sqrt = new Plaintext();
    encoder_->encode(mac_vec_sqrt, *enc_mac_sqrt);

    vector<uint64_t> mac_vec_cube(encoder_->slot_count(), mac_key_cube);
    Plaintext* enc_mac_cube = new Plaintext();
    encoder_->encode(mac_vec_cube, *enc_mac_cube);

    //Generate random_shares
    vector<uint64_t> op_shares_vec(num_rows, 0);
    vector<uint64_t> mac_op_shares_vec(num_rows, 0);
    vector<uint64_t> mac_ver_shares_vec(common_dim, 0);
    random_mod_p(prg, op_shares_vec.data(), num_rows, prime_mod);
    random_mod_p(prg, mac_op_shares_vec.data(), num_rows, prime_mod);
    random_mod_p(prg, mac_ver_shares_vec.data(), common_dim, prime_mod);

    Plaintext linear_0 = fc_preprocess_noise(op_shares_vec.data(), data, *encoder_);
    Plaintext linear_mac_0 = fc_preprocess_noise(mac_op_shares_vec.data(), data, *encoder_);
    Plaintext mac_ver_shares_0 = preprocess_vec_plain(mac_ver_shares_vec.data(), data, *encoder_);
    if (verbose)
      cout << "[Server] Generate shares" << endl;

    //Preprocess Matrix
    vector<uint64_t *> matrixd(num_rows);
    for (int i = 0; i < num_rows; i++) {
      matrixd[i] = new uint64_t[common_dim];
      for (int j = 0; j < common_dim; j++) {
        matrixd[i][j] = matrix[i][j];
      }
    }
    auto encoded_mat = preprocess_matrix(matrixd.data(), data, *encoder_);
    if (verbose)
      cout << "[Server] Preprocess Matrix" << endl;


    //Receive ciphertext from client
    Ciphertext input_ct;
    Ciphertext mac_input_ct;
    recv_ciphertext(io, context, input_ct);
    recv_ciphertext(io, context, mac_input_ct);
    if (verbose)
      cout << "[Server] Receive Ciphertext from client" << endl;

    //Add Shares
    evaluator_->add_plain_inplace(input_ct, input_pt);
    evaluator_->add_plain_inplace(mac_input_ct, mac_input_pt);

    //Compute FC component
    Ciphertext linear = fc_online(input_ct, encoded_mat, data, *evaluator_, *gal_keys_, *relin_keys_,
                               *zero_);
    Ciphertext linear_mac = fc_online(mac_input_ct, encoded_mat, data, *evaluator_, *gal_keys_, *relin_keys_,
                               *zero_);
    Ciphertext mac_ver_op_cube;
    Ciphertext mac_ver_op_sqrt;
    Ciphertext mac_ver_op;
    evaluator_->multiply_plain(input_ct, *enc_mac_cube, mac_ver_op_cube);
    evaluator_->multiply_plain(mac_input_ct, *enc_mac_sqrt, mac_ver_op_sqrt);
    evaluator_->sub(mac_ver_op_cube, mac_ver_op_sqrt, mac_ver_op);

    if (verbose)
      cout << "[Server] Compute FC" << endl;

    //Linear Share
    evaluator_->sub_plain_inplace(linear, linear_0);
    //Linear MAC Share
    evaluator_->sub_plain_inplace(linear_mac, linear_mac_0);
    //MAC Verification Share
    evaluator->sub_plain_inplace(mac_ver_op, mac_ver_shares_0);
    if (verbose)
      cout << "[Server] Generate Client shares" << endl;

    //Send ciphertexts to client
    send_ciphertext(io, linear);
    send_ciphertext(io, linear_mac);
    send_ciphertext(io, mac_ver_op);
    if (verbose)
      cout << "[Server] Send ciphertexts of shares to client" << endl;

    //Verify
    if(verify_output) {
      vector<uint64_t>  input_vec_1(common_dim);
      vector<uint64_t>  mac_input_vec_1(common_dim);
      //receive client input
      io->recv_data(input_vec_1.data(), sizeof(uint64_t)*common_dim);
      io->recv_data(mac_input_vec_1.data(), sizeof(uint64_t)*common_dim);
      for(int i=0; i<common_dim; i++) {
        input_vec_1[i] = (input_vec_1[i] + input_vec[i]) % prime_mod;
        mac_input_vec_1[i] = (mac_input_vec_1[i] + mac_input_vec[i]) % prime_mod;
      }

      //Compute FC
      auto result_actual_input = ideal_functionality(input_vec_1.data(), matrixd.data(), mod);
      auto result_actual_mac = ideal_functionality(mac_input_vec_1.data(), matrixd.data(), mod);

      vector<uint64_t> mac_ver_actual(common_dim, 0);
      for(int i=0; i<common_dim; i++) {
        mac_ver_actual[i] = (mod_mult(mac_key_cube, input_vec_1[i], mod) + (prime_mod - mod_mult(mac_key_sqrt, mac_input_vec_1[i], mod)))%prime_mod;
      }

      //receive client_shares
      vector<uint64_t> op_shares_vec_1(num_rows);
      vector<uint64_t> mac_op_shares_vec_1(num_rows);
      vector<uint64_t> mac_ver_shares_vec_1(common_dim);
      io->recv_data(op_shares_vec_1.data(), sizeof(uint64_t)*num_rows);
      io->recv_data(mac_op_shares_vec_1.data(), sizeof(uint64_t)*num_rows);
      io->recv_data(mac_ver_shares_vec_1.data(), sizeof(uint64_t)*common_dim);
      // reconstruct output
      for(int i=0; i<num_rows; i++) {
        op_shares_vec_1[i] = (op_shares_vec[i] + op_shares_vec_1[i])%prime_mod;
        mac_op_shares_vec_1[i] = (mac_op_shares_vec[i] + mac_op_shares_vec_1[i])%prime_mod;
      }

      for(int i=0; i<common_dim; i++) {
        mac_ver_shares_vec_1[i] = (mac_ver_shares_vec[i] + mac_ver_shares_vec_1[i])%prime_mod;
      }

      //check for equality
      int ctr=0;
      int mctr=0;
      int mvctr=0;
      for(int i=0;i<num_rows; i++) {
        if(result_actual_input[i] == op_shares_vec_1[i])
          ctr++;
        if(result_actual_mac[i] == mac_op_shares_vec_1[i])
          mctr++;
      }
      for(int i=0; i<common_dim; i++) {
        if(mac_ver_actual[i] == mac_ver_shares_vec_1[i]) {
            mvctr++;
        }
      }
      cout<<"Correct Shares: "<< ctr<<endl;
      cout<<"Correct Mac Shares: "<<mctr<<endl;
      cout<<"Correct Mac Verification Shares: "<<mvctr<<endl;
    }
  }
}
/*
void FCField::matrix_multiplication(int32_t num_rows, int32_t common_dim,
                                    int32_t num_cols,
                                    vector<vector<uint64_t>> &A,
                                    vector<vector<uint64_t>> &B,
                                    vector<vector<uint64_t>> &C,
                                    bool verify_output, bool verbose) {
  assert(num_cols == 1);
  data.filter_h = num_rows;
  data.filter_w = common_dim;
  data.image_size = common_dim;
  this->slot_count =
      min(max(8192, 2 * next_pow2(common_dim)), SEAL_POLY_MOD_DEGREE_MAX);
  configure();

  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  Ciphertext *zero_;
  if (slot_count > POLY_MOD_DEGREE) {
    generate_new_keys(party, io, slot_count, context, encryptor_, decryptor_,
                      evaluator_, encoder_, gal_keys_, zero_);
  } else {
    encryptor_ = this->encryptor;
    decryptor_ = this->decryptor;
    evaluator_ = this->evaluator;
    encoder_ = this->encoder;
    gal_keys_ = this->gal_keys;
    zero_ = this->zero;
  }

  if (party == BOB) {
    vector<uint64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = B[i][0];
    }
    if (verbose)
      cout << "[Client] Vector Generated" << endl;

    auto ct = preprocess_vec(vec.data(), data, *encryptor_, *encoder_);
    send_ciphertext(io, ct);
    if (verbose)
      cout << "[Client] Vector processed and sent" << endl;

    Ciphertext enc_result;
    recv_ciphertext(io, context, enc_result);
    auto HE_result = fc_postprocess(enc_result, data, *encoder_, *decryptor_);
    if (verbose)
      cout << "[Client] Result received and decrypted" << endl;

    for (int i = 0; i < num_rows; i++) {
      C[i][0] = HE_result[i];
    }
    if (verify_output)
      verify(&vec, nullptr, C);

    delete[] HE_result;
  } else // party == ALICE
  {
    vector<uint64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = B[i][0];
    }
    if (verbose)
      cout << "[Server] Vector Generated" << endl;
    vector<uint64_t *> matrix_mod_p(num_rows);
    vector<uint64_t *> matrix(num_rows);
    for (int i = 0; i < num_rows; i++) {
      matrix_mod_p[i] = new uint64_t[common_dim];
      matrix[i] = new uint64_t[common_dim];
      for (int j = 0; j < common_dim; j++) {
        matrix_mod_p[i][j] = neg_mod((int64_t)A[i][j], (int64_t)prime_mod);
        matrix[i][j] = A[i][j];
      }
    }
    if (verbose)
      cout << "[Server] Matrix generated" << endl;

    PRG prg;
    uint64_t *secret_share = new uint64_t[num_rows];
    random_mod_p(prg, secret_share, num_rows, prime_mod);

    Ciphertext enc_noise =
        fc_preprocess_noise(secret_share, data, *encryptor_, *encoder_);
    auto encoded_mat = preprocess_matrix(matrix_mod_p.data(), data, *encoder_);
    if (verbose)
      cout << "[Server] Matrix and noise processed" << endl;

    Ciphertext ct;
    recv_ciphertext(io, context, ct);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, ct, "before FC Online");
#endif

    auto HE_result = fc_online(ct, encoded_mat, data, *evaluator_, *gal_keys_,
                               *zero_, enc_noise);
/*
#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after FC Online");
#endif

    parms_id_type parms_id = HE_result.parms_id();
    shared_ptr<const SEALContext::ContextData> context_data =
        context->get_context_data(parms_id);
    flood_ciphertext(HE_result, context_data, SMUDGING_BITLEN);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after noise flooding");
#endif*/

    //evaluator_->mod_switch_to_next_inplace(HE_result);
/*
#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after mod-switch");
#endif

    send_ciphertext(io, HE_result);
    if (verbose)
      cout << "[Server] Result computed and sent" << endl;

    auto result = ideal_functionality(vec.data(), matrix.data());

    for (int i = 0; i < num_rows; i++) {
      C[i][0] = neg_mod((int64_t)result[i] - (int64_t)secret_share[i],
                        (int64_t)prime_mod);
    }
    if (verify_output)
      verify(&vec, &matrix, C);

    for (int i = 0; i < num_rows; i++) {
      delete[] matrix_mod_p[i];
      delete[] matrix[i];
    }
    delete[] secret_share;
  }
  if (slot_count > POLY_MOD_DEGREE) {
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_,
              zero_);
  }
}
*/
void FCField::verify(vector<uint64_t> *vec, vector<uint64_t *> *matrix,
                     vector<vector<uint64_t>> &C) {
  if (party == BOB) {
    io->send_data(vec->data(), data.filter_w * sizeof(uint64_t));
    io->flush();
    for (int i = 0; i < data.filter_h; i++) {
      io->send_data(C[i].data(), sizeof(uint64_t));
    }
  } else // party == ALICE
  {
    vector<uint64_t> vec_0(data.filter_w);
    io->recv_data(vec_0.data(), data.filter_w * sizeof(uint64_t));
    for (int i = 0; i < data.filter_w; i++) {
      vec_0[i] = (vec_0[i] + (*vec)[i]) % prime_mod;
    }/*
    auto result = ideal_functionality(vec_0.data(), matrix->data());

    vector<vector<uint64_t>> C_0(data.filter_h);
    for (int i = 0; i < data.filter_h; i++) {
      C_0[i].resize(1);
      io->recv_data(C_0[i].data(), sizeof(uint64_t));
      C_0[i][0] = (C_0[i][0] + C[i][0]) % prime_mod;
    }
    bool pass = true;
    for (int i = 0; i < data.filter_h; i++) {
      if (neg_mod(result[i], (int64_t)prime_mod) != (int64_t)C_0[i][0]) {
        pass = false;
      }
    }
    if (pass)
      cout << GREEN << "[Server] Successful Operation" << RESET << endl;
    else {
      cout << RED << "[Server] Failed Operation" << RESET << endl;
      cout << RED << "WARNING: The implementation assumes that the computation"
           << endl;
      cout << "performed locally by the server (on the model and its input "
              "share)"
           << endl;
      cout << "fits in a 64-bit integer. The failed operation could be a result"
           << endl;
      cout << "of overflowing the bound." << RESET << endl;
    }*/
  }
}
