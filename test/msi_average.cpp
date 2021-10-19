#include "emp-sh2pc/emp-sh2pc.h"
#include <cmath>

#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include "LinearLayer/defines-HE.h"
#include <thread>
#include "LinearLayer/utils-HE.h"
#define MAX_THREADS 8
using namespace emp;
using namespace std;

enum neural_net {
  NONE,
  MINIONN,
  CIFAR10
};

struct dimension {
  int N;
  int l;
  int b;
  int d;
};

neural_net choice_nn;
neural_net def_nn = NONE;

string address;

uint64_t prime_val = 17592060215297;
seal::Modulus mod(prime_val);

uint64_t prime_mod;
uint64_t moduloMask;
uint64_t moduloMidPt;
uint64_t avg_pool_const = 64;

uint64_t mac_key;
PRG prg;

NetIO *ioArr[MAX_THREADS];

uint64_t prime_field;

int l = 44;

typedef std::vector<uint64_t> uint64_1D;

template <typename T> vector<T> make_vector(size_t size) {
  return std::vector<T>(size);
}

template <typename T> T *make_array(size_t s1) { return new T[s1]; }

template <typename T> T *make_array(size_t s1, size_t s2) {
  return new T[s1 * s2];
}

template <typename T> T *make_array(size_t s1, size_t s2, size_t s3) {
  return new T[s1 * s2 * s3];
}

template <typename T>
T *make_array(size_t s1, size_t s2, size_t s3, size_t s4) {
  return new T[s1 * s2 * s3 * s4];
}

template <typename T>
T *make_array(size_t s1, size_t s2, size_t s3, size_t s4, size_t s5) {
  return new T[s1 * s2 * s3 * s4 * s5];
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

void div_floor(int64_t a, int64_t b, int64_t &quot, int64_t &rem) {
  assert(b > 0);
  int64_t q = a / b;
  int64_t r = a % b;
  int64_t corr = ((r != 0) && (r < 0));
  quot = q - corr;
  rem = (r + b) % b;
}

inline int64_t getSignedVal(uint64_t x) {
  assert(x < prime_mod);
  int64_t sx = x;
  if (x >= moduloMidPt)
    sx = x - prime_mod;
  return sx;
}

inline uint64_t getRingElt(int64_t x) { return ((uint64_t)x) & moduloMask; }

inline uint64_t PublicAdd(uint64_t x, uint64_t y) {
  assert((x < prime_mod) && (y < prime_mod));
  return (x + y) & moduloMask;
}

inline uint64_t PublicSub(uint64_t x, uint64_t y) {
  assert((x < prime_mod) && (y < prime_mod));
  return (x - y) & moduloMask;
}

inline uint64_t PublicMult(uint64_t x, uint64_t y) {
  assert((x < prime_mod) && (y < prime_mod));
  return (x * y) & moduloMask; // This works because its a two-power ring
}

inline bool PublicGT(uint64_t x, uint64_t y) {
  int64_t sx = getSignedVal(x);
  int64_t sy = getSignedVal(y);
  return (sx > sy);
}

inline bool PublicGTE(uint64_t x, uint64_t y) {
  int64_t sx = getSignedVal(x);
  int64_t sy = getSignedVal(y);
  return (sx >= sy);
}

inline bool PublicLT(uint64_t x, uint64_t y) {
  int64_t sx = getSignedVal(x);
  int64_t sy = getSignedVal(y);
  return (sx < sy);
}

inline bool PublicLTE(uint64_t x, uint64_t y) {
  int64_t sx = getSignedVal(x);
  int64_t sy = getSignedVal(y);
  return (sx <= sy);
}

uint64_t PublicDiv(uint64_t x, uint64_t y) {
  int64_t sx = getSignedVal(x);
  int64_t sy = getSignedVal(y);
  int64_t q, r;
  div_floor(sx, sy, q, r);
  return getRingElt(q);
}

uint64_t PublicMod(uint64_t x, uint64_t y) {
  int64_t sx = getSignedVal(x);
  int64_t sy = getSignedVal(y);
  int64_t q, r;
  div_floor(sx, sy, q, r);
  return r;
}

inline uint64_t PublicRShiftA(uint64_t x, uint64_t y) {
  assert((x < prime_mod) && (y < prime_mod));
  int64_t sx = getSignedVal(x);
  int64_t ans = sx >> y;
  return getRingElt(ans);
}

inline uint64_t PublicRShiftL(uint64_t x, uint64_t y) {
  assert((x < prime_mod) && (y < prime_mod));
  return (x >> y);
}

inline uint64_t PublicLShift(uint64_t x, uint64_t y) {
  assert((x < prime_mod) && (y < prime_mod));
  return (x << y) & moduloMask;
}

void AvgPool_pt(uint64_t N, uint64_t H, uint64_t W, uint64_t C, uint64_t ksizeH,
                uint64_t ksizeW, uint64_t zPadHLeft, uint64_t zPadHRight,
                uint64_t zPadWLeft, uint64_t zPadWRight, uint64_t strideH,
                uint64_t strideW, uint64_t N1, uint64_t imgH, uint64_t imgW,
                uint64_t C1,
                std::vector<std::vector<std::vector<uint64_1D>>> &inArr,
                std::vector<std::vector<std::vector<uint64_1D>>> &outArr) {
  uint64_t rows = (PublicMult((PublicMult((PublicMult(N, C)), H)), W));

  auto filterAvg = make_vector<uint64_t>(rows);

  uint64_t rowIdx = (int32_t)0;
  for (uint64_t n = (int32_t)0; n < N; n++) {
    for (uint64_t c = (int32_t)0; c < C; c++) {

      uint64_t leftTopCornerH = (PublicSub((int32_t)0, zPadHLeft));

      uint64_t extremeRightBottomCornerH =
          (PublicAdd((PublicSub(imgH, (int32_t)1)), zPadHRight));

      uint64_t ctH = (int32_t)0;
      while ((PublicLTE(
          (PublicSub((PublicAdd(leftTopCornerH, ksizeH)), (int32_t)1)),
          extremeRightBottomCornerH))) {

        uint64_t leftTopCornerW = (PublicSub((int32_t)0, zPadWLeft));

        uint64_t extremeRightBottomCornerW =
            (PublicAdd((PublicSub(imgW, (int32_t)1)), zPadWRight));

        uint64_t ctW = (int32_t)0;
        while ((PublicLTE(
            (PublicSub((PublicAdd(leftTopCornerW, ksizeW)), (int32_t)1)),
            extremeRightBottomCornerW))) {

          uint64_t curFilterSum = (int64_t)0;
          for (uint64_t fh = (int32_t)0; fh < ksizeH; fh++) {
            for (uint64_t fw = (int32_t)0; fw < ksizeW; fw++) {

              uint64_t curPosH = (PublicAdd(leftTopCornerH, fh));

              uint64_t curPosW = (PublicAdd(leftTopCornerW, fw));

              uint64_t temp = (int64_t)0;
              if ((((PublicLT(curPosH, (int32_t)0)) ||
                    (PublicGTE(curPosH, imgH))) ||
                   ((PublicLT(curPosW, (int32_t)0)) ||
                    (PublicGTE(curPosW, imgW))))) {
                temp = (int64_t)0;
              } else {
                temp = inArr[n][curPosH][curPosW][c];
              }
              curFilterSum = (PublicAdd(curFilterSum, temp));
            }
          }

          uint64_t ksizeH64 = ksizeH;

          uint64_t ksizeW64 = ksizeW;

          uint64_t filterSz64 = (PublicMult(ksizeH64, ksizeW64));

          uint64_t curFilterAvg = (PublicDiv(curFilterSum, filterSz64));
          filterAvg[rowIdx] = curFilterAvg;
          rowIdx = (PublicAdd(rowIdx, (int32_t)1));
          leftTopCornerW = (PublicAdd(leftTopCornerW, strideW));
          ctW = (PublicAdd(ctW, (int32_t)1));
        }

        leftTopCornerH = (PublicAdd(leftTopCornerH, strideH));
        ctH = (PublicAdd(ctH, (int32_t)1));
      }
    }
  }
  for (uint64_t n = (int32_t)0; n < N; n++) {
    for (uint64_t c = (int32_t)0; c < C; c++) {
      for (uint64_t h = (int32_t)0; h < H; h++) {
        for (uint64_t w = (int32_t)0; w < W; w++) {
          outArr[n][h][w][c] = filterAvg[(PublicAdd(
              (PublicAdd(
                  (PublicAdd(
                      (PublicMult((PublicMult((PublicMult(n, C)), H)), W)),
                      (PublicMult((PublicMult(c, H)), W)))),
                  (PublicMult(h, W)))),
              w))];
        }
      }
    }
  }
}

void parse_arguments(int argc, char**arg, int *party, int *port, int *bitlen) {
  *party = atoi (arg[1]);
   address = arg[2];
	*port = atoi (arg[3]);
  if(argc < 5) {
    *bitlen = l;
  } else {
    *bitlen = atoi(arg[4]);
  }

  if(argc < 6) {
    choice_nn =def_nn;
  } else {
    choice_nn = neural_net(atoi (arg[5]));
  }

  prime_mod = (*bitlen == 64 ? 0ULL : 1ULL << *bitlen);
  moduloMask = prime_mod - 1;
  moduloMidPt = prime_mod / 2;
}

int main(int argc, char** argv) {
  srand(time(NULL));
  int port, party, nrelu, bitlen;
  //Parse input arguments and configure parameters
	parse_arguments(argc, argv, &party, &port, &bitlen);

  ioArr[0] = new NetIO(party==ALICE ? nullptr : address.c_str(), port);

  //Prepare and share inputs
  uint64_t layers_count=2;

  uint64_t *inputs[layers_count], *inputs_mac[layers_count], *outputs[layers_count], *outputs_mac[layers_count];
  uint64_t *prepared_input[layers_count], *prepared_input_mac[layers_count];
  uint64_t *send_inputs[layers_count], *send_input_mac[layers_count];

  dimension input_dim[layers_count], output_dim[layers_count];

  if(choice_nn == MINIONN) {
    input_dim[0].N = 1; input_dim[0].l = 24, input_dim[0].b = 24, input_dim[0].d = 16;
    input_dim[1].N = 1; input_dim[1].l = 8, input_dim[1].b = 8, input_dim[1].d = 16;

    output_dim[0].N = 1; output_dim[0].l = 12, output_dim[0].b = 12, output_dim[0].d = 16;
    output_dim[1].N = 1; output_dim[1].l = 4, output_dim[1].b = 4, output_dim[1].d = 16;
  } else {
    input_dim[0].N = 1; input_dim[0].l = 32, input_dim[0].b = 32, input_dim[0].d = 64;
    input_dim[1].N = 1; input_dim[1].l = 16, input_dim[1].b = 16, input_dim[1].d = 64;

    output_dim[0].N = 1; output_dim[0].l = 16, output_dim[0].b = 16, output_dim[0].d = 64;
    output_dim[1].N = 1; output_dim[1].l = 16, output_dim[1].b = 16, output_dim[1].d = 64;
  }

  for(int i=0; i< layers_count; i++) {
    inputs[i] = make_array<uint64_t>((int32_t)input_dim[i].N, (int32_t)input_dim[i].l, (int32_t)input_dim[i].b, (int32_t)input_dim[i].d);
    inputs_mac[i] = make_array<uint64_t>((int32_t)input_dim[i].N, (int32_t)input_dim[i].l, (int32_t)input_dim[i].b, (int32_t)input_dim[i].d);

    outputs[i] = make_array<uint64_t>((int32_t)output_dim[i].N, (int32_t)output_dim[i].l, (int32_t)output_dim[i].b, (int32_t)output_dim[i].d);
    outputs_mac[i] = make_array<uint64_t>((int32_t)output_dim[i].N, (int32_t)output_dim[i].l, (int32_t)output_dim[i].b, (int32_t)output_dim[i].d);
  }

  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint64_t> distr;

  if(party == ALICE) {
    prg.random_data(&mac_key, 8);
    mac_key %= prime_val;

    for(int i=0; i<layers_count; i++) {
      prepared_input[i] = make_array<uint64_t>((int32_t)input_dim[i].N, (int32_t)input_dim[i].l, (int32_t)input_dim[i].b, (int32_t)input_dim[i].d);
      prepared_input_mac[i] = make_array<uint64_t>((int32_t)input_dim[i].N, (int32_t)input_dim[i].l, (int32_t)input_dim[i].b, (int32_t)input_dim[i].d);
      send_inputs[i] = make_array<uint64_t>((int32_t)input_dim[i].N, (int32_t)input_dim[i].l, (int32_t)input_dim[i].b, (int32_t)input_dim[i].d);
      send_input_mac[i] = make_array<uint64_t>((int32_t)input_dim[i].N, (int32_t)input_dim[i].l, (int32_t)input_dim[i].b, (int32_t)input_dim[i].d);
      int arr_size = input_dim[i].N * input_dim[i].l * input_dim[i].b * input_dim[i].d;
      random_mod_p(prg, prepared_input[i], arr_size, prime_val);
      for(int j=0; j< arr_size; j++) {
        prepared_input_mac[i][j] = mod_mult(mac_key,prepared_input[i][j]);
      }
      random_mod_p(prg, inputs[i], arr_size, prime_val);
      random_mod_p(prg, inputs_mac[i], arr_size, prime_val);
      for(int j=0; j<arr_size; j++) {
        send_inputs[i][j] = (prepared_input[i][j] - inputs[i][j])%prime_val;
        send_input_mac[i][j] = (prepared_input_mac[i][j] - inputs_mac[i][j])%prime_val;
      }
      ioArr[0]->send_data(send_inputs[i], sizeof(uint64_t) * arr_size);
      ioArr[0]->send_data(send_input_mac[i], sizeof(uint64_t) * arr_size);
    }
  } else {
    for(int i=0; i<layers_count; i++) {
      int arr_size = input_dim[i].N * input_dim[i].l * input_dim[i].b * input_dim[i].d;
      ioArr[0]->recv_data(inputs[i], sizeof(uint64_t)* arr_size);
      ioArr[0]->recv_data(inputs_mac[i], sizeof(uint64_t)* arr_size);
    }
  }


  //Performance Result
  std::vector<std::vector<std::vector<std::vector<uint64_t>>>> inVec[layers_count];
  std::vector<std::vector<std::vector<std::vector<uint64_t>>>> inVecMac[layers_count];
  std::vector<std::vector<std::vector<std::vector<uint64_t>>>> outVec[layers_count];
  std::vector<std::vector<std::vector<std::vector<uint64_t>>>> outVecMac[layers_count];

  for(int i=0; i<layers_count; i++) {
   inVec[i].resize(input_dim[i].N, std::vector<std::vector<std::vector<uint64_t>>>(
                         input_dim[i].l, std::vector<std::vector<uint64_t>>(
                                   input_dim[i].b, std::vector<uint64_t>(input_dim[i].d, 0))));
   inVecMac[i].resize(input_dim[i].N, std::vector<std::vector<std::vector<uint64_t>>>(
                          input_dim[i].l, std::vector<std::vector<uint64_t>>(
                                    input_dim[i].b, std::vector<uint64_t>(input_dim[i].d, 0))));
   outVec[i].resize(output_dim[i].N, std::vector<std::vector<std::vector<uint64_t>>>(
                          output_dim[i].l, std::vector<std::vector<uint64_t>>(
                                    output_dim[i].b, std::vector<uint64_t>(output_dim[i].d, 0))));
   outVecMac[i].resize(output_dim[i].N, std::vector<std::vector<std::vector<uint64_t>>>(
                         output_dim[i].l, std::vector<std::vector<uint64_t>>(
                                   output_dim[i].b, std::vector<uint64_t>(output_dim[i].d, 0))));
    for(int j=0; j<input_dim[i].N; j++) {
      for(int k=0; k<input_dim[i].l; k++) {
        for(int l=0; l<input_dim[i].b; l++) {
          for(int m=0; m<input_dim[i].d; m++) {
            inVec[i][j][k][l][m] = inputs[i][(j) * (input_dim[i].l) * (input_dim[i].b) * (input_dim[i].d) + (k) * (input_dim[i].b) * (input_dim[i].d) + (l) * (input_dim[i].d) + (m)];
            inVecMac[i][j][k][l][m] = (*((inputs_mac[i]) + (j) * (input_dim[i].l) * (input_dim[i].b) * (input_dim[i].d) + (k) * (input_dim[i].b) * (input_dim[i].d) + (l) * (input_dim[i].d) + (m)));
          }
        }
      }
    }
  }


  auto start = clock_start();
  if(choice_nn == MINIONN) {
    AvgPool_pt((int32_t) output_dim[0].N, (int32_t) output_dim[0].l, (int32_t) output_dim[0].b, (int32_t) output_dim[0].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 0,
                    (int32_t) 0, (int32_t) 0, (int32_t) 2,
                    (int32_t) 2, (int32_t) input_dim[0].N, (int32_t) input_dim[0].l, (int32_t) input_dim[0].b,
                    (int32_t) input_dim[0].d,
                    inVec[0], outVec[0]);

    AvgPool_pt((int32_t) output_dim[0].N, (int32_t) output_dim[0].l, (int32_t) output_dim[0].b, (int32_t) output_dim[0].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 0,
                    (int32_t) 0, (int32_t) 0, (int32_t) 2,
                    (int32_t) 2, (int32_t) input_dim[0].N, (int32_t) input_dim[0].l, (int32_t) input_dim[0].b,
                    (int32_t) input_dim[0].d,
                    inVecMac[0], outVecMac[0]);
    AvgPool_pt((int32_t) output_dim[1].N, (int32_t) output_dim[1].l, (int32_t) output_dim[1].b, (int32_t) output_dim[1].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 0,
                    (int32_t) 0, (int32_t) 0, (int32_t) 2,
                    (int32_t) 2, (int32_t) input_dim[1].N, (int32_t) input_dim[1].l, (int32_t) input_dim[1].b,
                    (int32_t) input_dim[1].d,
                    inVec[1], outVecMac[1]);
    AvgPool_pt((int32_t) output_dim[1].N, (int32_t) output_dim[1].l, (int32_t) output_dim[1].b, (int32_t) output_dim[1].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 0,
                    (int32_t) 0, (int32_t) 0, (int32_t) 2,
                    (int32_t) 2, (int32_t) input_dim[1].N, (int32_t) input_dim[1].l, (int32_t) input_dim[1].b,
                    (int32_t) input_dim[1].d,
                    inVecMac[1], outVecMac[1]);
  } else {
    AvgPool_pt((int32_t) output_dim[0].N, (int32_t) output_dim[0].l, (int32_t) output_dim[0].b, (int32_t) output_dim[0].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 0,
                    (int32_t) 0, (int32_t) 0, (int32_t) 2,
                    (int32_t) 2, (int32_t) input_dim[0].N, (int32_t) input_dim[0].l, (int32_t) input_dim[0].b,
                    (int32_t) input_dim[0].d,
                    inVec[0], outVec[0]);

    AvgPool_pt((int32_t) output_dim[0].N, (int32_t) output_dim[0].l, (int32_t) output_dim[0].b, (int32_t) output_dim[0].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 0,
                    (int32_t) 0, (int32_t) 0, (int32_t) 2,
                    (int32_t) 2, (int32_t) input_dim[0].N, (int32_t) input_dim[0].l, (int32_t) input_dim[0].b,
                    (int32_t) input_dim[0].d,
                    inVecMac[0], outVecMac[0]);
    AvgPool_pt((int32_t) output_dim[1].N, (int32_t) output_dim[1].l, (int32_t) output_dim[1].b, (int32_t) output_dim[1].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 1,
                    (int32_t) 0, (int32_t) 1, (int32_t) 1,
                    (int32_t) 1, (int32_t) input_dim[1].N, (int32_t) input_dim[1].l, (int32_t) input_dim[1].b,
                    (int32_t) input_dim[1].d,
                    inVec[1], outVec[1]);
    AvgPool_pt((int32_t) output_dim[1].N, (int32_t) output_dim[1].l, (int32_t) output_dim[1].b, (int32_t) output_dim[1].d, (int32_t) 2,
                    (int32_t) 2, (int32_t) 0, (int32_t) 1,
                    (int32_t) 0, (int32_t) 1, (int32_t) 1,
                    (int32_t) 1, (int32_t) input_dim[1].N, (int32_t) input_dim[1].l, (int32_t) input_dim[1].b,
                    (int32_t) input_dim[1].d,
                    inVecMac[1], outVecMac[1]);
  }
  long long t = time_from(start);
  cout << "######################Performance#######################" <<endl;
  cout<<"Time Taken: "<<t<<" mus"<<endl;
  cout<<"Sent Data (MB): "<<0<<endl;
  cout << "########################################################" <<endl;

}
