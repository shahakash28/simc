#include "emp-sh2pc/emp-sh2pc.h"
#include <cmath>

#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include <thread>
#define MAX_THREADS 8
using namespace emp;
using namespace std;


int num_threads = 8;

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

enum neural_net {
  NONE,
  MINIONN,
  CIFAR10
};

neural_net choice_nn;
int choose_relu;
uint64_t start_comm[MAX_THREADS];
uint64_t comm_sent = 0;
NetIO *ioArr[MAX_THREADS];
uint64_t prime_mod = 17592060215297;
seal::Modulus mod(prime_mod);
int port = 32000, def_nrelu = 1<<20, l = 44;
neural_net def_nn = NONE;
string address;
bool run_all = false;
uint64_t mac_key;
PRG prg;

bool verify = false;
int MINIONN_RELUS[] = { 16*576, 16*64, 100*1
};

int CIFAR10_RELUS[] = { 64*1024, 64*1024, 64*256, 64*256, 64*64, 64*64, 16*64
};

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

void msi_relu_6(int party, NetIO* io, uint64_t* inputs, int nrelu, int bitlen, uint64_t* ip_ss, uint64_t* op_ss, uint64_t* op_mss) {
  //Public prime values
  Integer p(bitlen + 1, prime_mod, PUBLIC);
  Integer p_mod2(bitlen, prime_mod/2, PUBLIC);
  Integer zero(bitlen, 0, PUBLIC);
  Integer six(bitlen, 6, PUBLIC);

  //Assign Inputs
  Integer *X = new Integer[nrelu];
  for(int i = 0; i < nrelu; ++i)
    X[i] = Integer(bitlen+1, inputs[i], ALICE);
  Integer *Y = new Integer[nrelu];
  for(int i = 0; i < nrelu; ++i)
    Y[i] = Integer(bitlen+1, inputs[i], BOB);

  Integer *S = new Integer[nrelu];
  Integer *T = new Integer[nrelu];

  //Check if Bob's share is < p
  Bit res[nrelu];
  for(int i=0; i < nrelu; ++i)
    res[i] = Y[i] > p;

  for(int i=0; i < nrelu; ++i) {
    //Perform mod p
    Integer s0 = X[i];
    //s0.resize(s0.size()+1);

    Integer s1 = Y[i];
   //s1.resize(s1.size()+1);

    Integer sum = s0 + s1;

    Integer mod_p_val = sum - p;

    Bit borrow_bit = mod_p_val[mod_p_val.size()-1];

    Integer s = mod_p_val.select(borrow_bit, sum);

    S[i] = s;

    //Perform RELU
    Integer p2_minus_s = p_mod2-s;

    Bit is_negative = p2_minus_s[p2_minus_s.size()-1];

    Integer relu_s = s.select(is_negative, zero);

    Integer six_minus_res = six - relu_s;
    Bit is_greater_than_six = six_minus_res[six_minus_res.size()-1];

    Integer res = relu_s.select(is_greater_than_six, six);

    T[i] = res;
  }

  int pack_size = ceil(nrelu*bitlen*bitlen*2.0/(8*sizeof(uint64_t)));
  int batch_size = nrelu*bitlen*2;

  uint64_t *ip_cts = (uint64_t *)malloc(nrelu*bitlen*2*sizeof(uint64_t));

  uint64_t *op_cts = (uint64_t *)malloc(nrelu*bitlen*2*sizeof(uint64_t));

  uint64_t *op_mcts = (uint64_t *)malloc(nrelu*bitlen*2*sizeof(uint64_t));

  uint64_t *ip_pack_table = (uint64_t *)malloc(pack_size*sizeof(uint64_t));
  uint64_t *op_pack_table = (uint64_t *)malloc(pack_size*sizeof(uint64_t));
  uint64_t *opm_pack_table = (uint64_t *)malloc(pack_size*sizeof(uint64_t));

  if(party == ALICE) {

    create_ciphertexts(S, delta_used, ip_cts, ip_ss, bitlen, nrelu, mac_key, 1);
    create_ciphertexts(T, delta_used, op_cts, op_ss, bitlen, nrelu, 1, 0);
    create_ciphertexts(T, delta_used, op_mcts, op_mss, bitlen, nrelu, mac_key, 1);

    pack_decryption_table(ip_pack_table, ip_cts, pack_size, batch_size, bitlen);
    pack_decryption_table(op_pack_table, op_cts, pack_size, batch_size, bitlen);
    pack_decryption_table(opm_pack_table, op_mcts, pack_size, batch_size, bitlen);

    //cout<<"First element (meth):"<<ip_pack_table[0]<<endl;
    io->send_data(ip_pack_table, sizeof(uint64_t) * pack_size);
    io->send_data(op_pack_table, sizeof(uint64_t) * pack_size);
    io->send_data(opm_pack_table, sizeof(uint64_t) * pack_size);
  } else {
    io->recv_data(ip_pack_table, sizeof(uint64_t) * pack_size);
    io->recv_data(op_pack_table, sizeof(uint64_t) * pack_size);
    io->recv_data(opm_pack_table, sizeof(uint64_t) * pack_size);

    unpack_decryption_table(ip_pack_table, ip_cts, pack_size, batch_size, bitlen);
    unpack_decryption_table(op_pack_table, op_cts, pack_size, batch_size, bitlen);
    unpack_decryption_table(opm_pack_table, op_mcts, pack_size, batch_size, bitlen);
    //cout<<"First element (meth):"<<ip_pack_table[0]<<endl;

    decrypt_ciphertexts(S, ip_cts, ip_ss, bitlen, nrelu, 1);
    decrypt_ciphertexts(T, op_cts, op_ss, bitlen, nrelu, 0);
    decrypt_ciphertexts(T, op_mcts, op_mss, bitlen, nrelu, 1);
  }
}

void msi_relu(int party, NetIO* io, uint64_t inputs[], int nrelu, int bitlen, uint64_t* ip_ss, uint64_t* op_ss, uint64_t* op_mss) {
  uint64_t comm_sent;
  //Public prime values
  Integer p(bitlen + 1, prime_mod, PUBLIC);
  Integer p_mod2(bitlen, prime_mod/2, PUBLIC);
  Integer zero(bitlen, 0, PUBLIC);

  //Assign Inputs
  Integer *X = new Integer[nrelu];
  for(int i = 0; i < nrelu; ++i)
    X[i] = Integer(bitlen+1, inputs[i], ALICE);
  Integer *Y = new Integer[nrelu];
  for(int i = 0; i < nrelu; ++i)
    Y[i] = Integer(bitlen+1, inputs[i], BOB);

  Integer *S = new Integer[nrelu];
  Integer *T = new Integer[nrelu];

  //Check if Bob's share is < p
  Bit res[nrelu];
  for(int i=0; i < nrelu; ++i)
    res[i] = Y[i] > p;

  for(int i=0; i < nrelu; ++i) {
    //Perform mod p
    Integer s0 = X[i];
    //s0.resize(s0.size()+1);

    Integer s1 = Y[i];
   //s1.resize(s1.size()+1);

    Integer sum = s0 + s1;

    Integer mod_p_val = sum - p;

    Bit borrow_bit = mod_p_val[mod_p_val.size()-1];

    Integer s = mod_p_val.select(borrow_bit, sum);

    S[i] = s;

    //Perform RELU
    Integer p2_minus_s = p_mod2-s;

    Bit is_negative = p2_minus_s[p2_minus_s.size()-1];

    Integer relu_s = s.select(is_negative, zero);

    T[i] = relu_s;
  }

  int pack_size = ceil(nrelu*bitlen*bitlen*2.0/(8*sizeof(uint64_t)));
  int batch_size = nrelu*bitlen*2;

  uint64_t *ip_cts = (uint64_t *)malloc(nrelu*bitlen*2*sizeof(uint64_t));

  uint64_t *op_cts = (uint64_t *)malloc(nrelu*bitlen*2*sizeof(uint64_t));

  uint64_t *op_mcts = (uint64_t *)malloc(nrelu*bitlen*2*sizeof(uint64_t));

  uint64_t *ip_pack_table = (uint64_t *)malloc(pack_size*sizeof(uint64_t));
  uint64_t *op_pack_table = (uint64_t *)malloc(pack_size*sizeof(uint64_t));
  uint64_t *opm_pack_table = (uint64_t *)malloc(pack_size*sizeof(uint64_t));

  if(party == ALICE) {

    create_ciphertexts(S, delta_used, ip_cts, ip_ss, bitlen, nrelu, mac_key, 1);
    create_ciphertexts(T, delta_used, op_cts, op_ss, bitlen, nrelu, 1, 0);
    create_ciphertexts(T, delta_used, op_mcts, op_mss, bitlen, nrelu, mac_key, 1);

    pack_decryption_table(ip_pack_table, ip_cts, pack_size, batch_size, bitlen);
    pack_decryption_table(op_pack_table, op_cts, pack_size, batch_size, bitlen);
    pack_decryption_table(opm_pack_table, op_mcts, pack_size, batch_size, bitlen);

    //cout<<"First element (meth):"<<ip_pack_table[0]<<endl;
    io->send_data(ip_pack_table, sizeof(uint64_t) * pack_size);
    io->send_data(op_pack_table, sizeof(uint64_t) * pack_size);
    io->send_data(opm_pack_table, sizeof(uint64_t) * pack_size);
  } else {
    io->recv_data(ip_pack_table, sizeof(uint64_t) * pack_size);
    io->recv_data(op_pack_table, sizeof(uint64_t) * pack_size);
    io->recv_data(opm_pack_table, sizeof(uint64_t) * pack_size);

    unpack_decryption_table(ip_pack_table, ip_cts, pack_size, batch_size, bitlen);
    unpack_decryption_table(op_pack_table, op_cts, pack_size, batch_size, bitlen);
    unpack_decryption_table(opm_pack_table, op_mcts, pack_size, batch_size, bitlen);
    //cout<<"First element (meth):"<<ip_pack_table[0]<<endl;

    decrypt_ciphertexts(S, ip_cts, ip_ss, bitlen, nrelu, 1);
    decrypt_ciphertexts(T, op_cts, op_ss, bitlen, nrelu, 0);
    decrypt_ciphertexts(T, op_mcts, op_mss, bitlen, nrelu, 1);
  }
}

void parse_arguments(int argc, char**arg, int *party, int *port, int *bitlen, int *nrelu) {
  *party = atoi (arg[1]);
   address = arg[2];
	*port = atoi (arg[3]);
  if(argc < 5) {
    *bitlen = l;
  } else {
    *bitlen = atoi(arg[4]);
  }

  if(argc < 6) {
    choose_relu = 0;
  } else {
    choose_relu = atoi(arg[5]);
  }

  if(argc < 7) {
    *nrelu =  def_nrelu;
  } else {
    *nrelu = atoi(arg[6]);
  }

  if(argc < 8) {
    num_threads = 8;
  } else {
    num_threads = atoi(arg[7]);
  }
}

void thread_process(int tid, int party, int choice, uint64_t* inputs, int nrelu, int bitlen, uint64_t* ip_ss, uint64_t* op_ss, uint64_t* op_mss) {
  setup_semi_honest(ioArr[tid], party);

  uint64_t nr_per_thread = nrelu/num_threads;
  uint64_t r = nrelu % num_threads;
  uint64_t actual_per_thread;
  if(tid == num_threads-1)
    actual_per_thread = nr_per_thread + r;
  else
    actual_per_thread = nr_per_thread;

  uint64_t offset = tid*nr_per_thread;

  if(choice == 0) {
    msi_relu_6(party, ioArr[tid], inputs+offset, actual_per_thread, bitlen, ip_ss+offset, op_ss+offset, op_mss+offset);
  } else {
    msi_relu(party, ioArr[tid], inputs+offset, actual_per_thread, bitlen, ip_ss+offset, op_ss+offset, op_mss+offset);
  }

  ioArr[tid]->flush();
  finalize_semi_honest();
}

int main(int argc, char** argv) {
  srand(time(NULL));
  int port, party, nrelu, bitlen;
  //Parse input arguments and configure parameters
	parse_arguments(argc, argv, &party, &port, &bitlen, &nrelu);
  cout<<"Running Microbenchmarks ..."<<endl;
  cout << "=====================Configuration======================" << endl;
  cout<<"Party Id: "<< party<<" - Server IP Address: "<< address <<" - Port: "<<port<<" - NRelu: "<<nrelu<<" - Bitlen: "<<bitlen<<" - Choice RELU: "<<choose_relu<<" - #Threads: "<<num_threads<<endl;
  cout << "========================================================" << endl;
  //Prepare Inputs
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint64_t> distr;

  uint64_t* inputs=(uint64_t *)malloc(nrelu*sizeof(uint64_t));
  for(int i = 0; i < nrelu; ++i)
    inputs[i] = distr(eng)%prime_mod;

  uint64_t *ip_ss = (uint64_t *)malloc(nrelu*sizeof(uint64_t));
  uint64_t *op_ss = (uint64_t *)malloc(nrelu*sizeof(uint64_t));
  uint64_t *op_mss = (uint64_t *)malloc(nrelu*sizeof(uint64_t));

  for(int i=0; i <num_threads; i++) {
    ioArr[i] = new NetIO(party==ALICE ? nullptr : address.c_str(), port+i);
  }

  //Communication Initialization
  for(int i=0; i<num_threads; i++)
    start_comm[i] = ioArr[i]->counter;

  //Time Begin
  auto start = clock_start();

  if(party == ALICE) {
    prg.random_data(&mac_key, 8);
    mac_key %= prime_mod;
  }

  std::thread relu_threads[num_threads];
  for(int i=0; i<num_threads; i++) {
    relu_threads[i] = std::thread(thread_process, i, party, choose_relu, inputs, nrelu, bitlen, ip_ss, op_ss, op_mss);
  }

  //Join
  for(int i=0; i<num_threads; i++) {
    relu_threads[i].join();
  }
  //Time End
  long long t = time_from(start);
  cout << "######################Performance#######################" <<endl;
  cout<<"Time Taken: "<<t<<" mus"<<endl;
  //Calculate Communication
  comm_sent = 0;
  for(int i=0; i<num_threads; i++) {
    comm_sent += (ioArr[i]->counter-start_comm[i]);
  }

  cout<<"Sent Data (Bytes): "<<comm_sent<<endl;
  comm_sent = comm_sent>>10;
  cout<<"Sent Data (KB): "<<comm_sent<<endl;
  comm_sent = comm_sent>>10;
  cout<<"Sent Data (MB): "<<comm_sent<<endl;
  cout << "########################################################" <<endl;


  //cout<<"nrelu: "<<nrelu<<endl;
  //Test Protocol
  if(verify) {
    ioArr[0] = new NetIO(party==ALICE ? nullptr : address.c_str(), port);
    if(party == BOB) {
      ioArr[0]->send_data(inputs, sizeof(uint64_t) * nrelu);
      ioArr[0]->send_data(ip_ss, sizeof(uint64_t) * nrelu);
      ioArr[0]->send_data(op_ss, sizeof(uint64_t) * nrelu);
      ioArr[0]->send_data(op_mss, sizeof(uint64_t) * nrelu);
    } else {
      uint64_t inputs_1[nrelu];
      uint64_t inputs_res[nrelu];
      uint64_t relu_res[nrelu];

      uint64_t *ip_ssc = (uint64_t *)malloc(nrelu*sizeof(uint64_t));
      uint64_t *op_ssc = (uint64_t *)malloc(nrelu*sizeof(uint64_t));
      uint64_t *op_mssc = (uint64_t *)malloc(nrelu*sizeof(uint64_t));

      ioArr[0]->recv_data(inputs_1, sizeof(uint64_t) * nrelu);
      ioArr[0]->recv_data(ip_ssc, sizeof(uint64_t) * nrelu);
      ioArr[0]->recv_data(op_ssc, sizeof(uint64_t) * nrelu);
      ioArr[0]->recv_data(op_mssc, sizeof(uint64_t) * nrelu);
      for(int i=0; i< nrelu; i++) {
        inputs_res[i] = (inputs_1[i] + inputs[i])%prime_mod;
        if(inputs_res[i] > prime_mod/2) {
          relu_res[i] = 0;
        } else {
          if(inputs_res[i] > 6)
            relu_res[i] = 6;
          else
            relu_res[i] = inputs_res[i];
        }
      }

      uint64_t ip_shares, ip_corr, op_shares, op_corr, opm_shares, opm_corr;
      uint64_t ctr_ip, ctr_op, ctr_opm=0;

      for(int i=0; i<nrelu; i++) {
        ip_shares = (ip_ss[i]+ip_ssc[i])%prime_mod;
        ip_corr = mod_mult(mac_key,inputs_res[i]);
        if(ip_shares == ip_corr)
          ctr_ip++;
        else {
          cout<<"Index: "<<i<<endl;
          break;
        }

        op_shares = (op_ss[i]+op_ssc[i])%prime_mod;
        if(op_shares == relu_res[i])
            ctr_op++;

        opm_shares = (op_mss[i] + op_mssc[i])%prime_mod;
        opm_corr = mod_mult(mac_key,relu_res[i]);
        if(opm_shares == opm_corr)
          ctr_opm++;
      }
      cout << "**********************Verification**********************" <<endl;
      cout<<"Correct Input Macs: "<< ctr_ip<<endl;
      cout<<"Correct Outputs: "<< ctr_op<<endl;
      cout<<"Correct Output Macs: "<< ctr_opm<<endl;
      cout << "********************************************************" <<endl;
    }
  }
  //Performance Result

}
