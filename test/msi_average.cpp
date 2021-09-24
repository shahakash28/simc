#include "emp-sh2pc/emp-sh2pc.h"
#include <cmath>

#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include <thread>
#define MAX_THREADS 8
using namespace emp;
using namespace std;

enum neural_net {
  NONE,
  MINIONN,
  CIFAR10
};

extern uint64_t prime_mod;
extern uint64_t moduloMask;

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
    choice_nn =def_nn;
  } else {
    choice_nn = neural_net(atoi (arg[5]));
  }
}

int main(int argc, char** argv) {
  srand(time(NULL));
  int port, party, nrelu, bitlen;
  //Parse input arguments and configure parameters
	parse_arguments(argc, argv, &party, &port, &bitlen, &nrelu);
  //Performance Result
}
