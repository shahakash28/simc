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

#include "LinearLayer/conv-field.h"
#include "LinearLayer/defines-HE.h"

using namespace std;
using namespace seal;
using namespace emp;

uint64_t prime_mod = PLAINTEXT_MODULUS;

enum neural_net {
  NONE,
  MINIONN,
  CIFAR10
};
neural_net choice_nn;
neural_net def_nn = NONE;

long long total_time = 0;

int party = 0;
int bitlength = 44;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int image_h = 28;
int image_w = 28;
int inp_chans = 1;
int out_chans = 16;
int filter_h = 5;
int filter_w = 5;
int stride_h = 1;
int stride_w = 1;

int stride = 1;
int filter_precision = 12;
int pad_l = 0;
int pad_r = 0;

seal::Modulus mod(prime_mod);

void Conv(ConvField &he_conv, int32_t H, int32_t CI, int32_t FH, int32_t CO,
          int32_t zPadHLeft, int32_t zPadHRight, int32_t strideH) {
  int newH = 1 + (H + zPadHLeft + zPadHRight - FH) / strideH;
  int N = 1;
  int W = H;
  int FW = FH;
  int zPadWLeft = zPadHLeft;
  int zPadWRight = zPadHRight;
  int strideW = strideH;
  int newW = newH;
  vector<vector<vector<vector<uint64_t>>>> inputArr(N);
  vector<vector<vector<vector<uint64_t>>>> filterArr(FH);
  vector<vector<vector<vector<uint64_t>>>> outArr(N);

  PRG prg;
  for (int i = 0; i < N; i++) {
    outArr[i].resize(newH);
    for (int j = 0; j < newH; j++) {
      outArr[i][j].resize(newW);
      for (int k = 0; k < newW; k++) {
        outArr[i][j][k].resize(CO);
      }
    }
  }
  if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
      filterArr[i].resize(FW);
      for (int j = 0; j < FW; j++) {
        filterArr[i][j].resize(CI);
        for (int k = 0; k < CI; k++) {
          filterArr[i][j][k].resize(CO);
          prg.random_data(filterArr[i][j][k].data(), CO * sizeof(uint64_t));
          for (int h = 0; h < CO; h++) {
            filterArr[i][j][k][h] =
                ((int64_t)filterArr[i][j][k][h]) >> (64 - filter_precision);
          }
        }
      }
    }
  }
  for (int i = 0; i < N; i++) {
    inputArr[i].resize(H);
    for (int j = 0; j < H; j++) {
      inputArr[i][j].resize(W);
      for (int k = 0; k < W; k++) {
        inputArr[i][j][k].resize(CI);
        random_mod_p(prg, inputArr[i][j][k].data(), CI, prime_mod);
      }
    }
  }
  uint64_t comm_start = he_conv.io->counter;

  he_conv.convolution(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                      zPadWRight, strideH, strideW, inputArr, filterArr, outArr,
                      true, true);
  uint64_t comm_end = he_conv.io->counter;
  cout << "Total Comm: " << (comm_end - comm_start) / (1.0 * (1ULL << 20))
       << endl;
}

void Conv_First(ConvField &he_conv, int32_t H, int32_t W, int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t strideH,
  int32_t strideW, bool pad_valid) {
  int N = 1;
  vector<vector<vector<vector<uint64_t>>>> inputArr(N);
  vector<vector<vector<vector<uint64_t>>>> filterArr(FH);

  PRG prg;
  cout<<"Party "<< party<<endl;
  if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
      filterArr[i].resize(FW);
      for (int j = 0; j < FW; j++) {
        filterArr[i][j].resize(CI);
        for (int k = 0; k < CI; k++) {
          filterArr[i][j][k].resize(CO);
          random_mod_p(prg, filterArr[i][j][k].data(), CO, prime_mod);
        }
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      inputArr[i].resize(H);
      for (int j = 0; j < H; j++) {
        inputArr[i][j].resize(W);
        for (int k = 0; k < W; k++) {
          inputArr[i][j][k].resize(CI);
          random_mod_p(prg, inputArr[i][j][k].data(), CI, prime_mod);
        }
      }
    }
  }

  auto start = clock_start();

  he_conv.convolution_first(H, W, CI, FH, FW, CO, strideH, strideW, pad_valid, inputArr, filterArr,
                      false, true);
  long long t = time_from(start);
  total_time += t;
}

void Conv_Gen(ConvField &he_conv, int32_t H, int32_t W, int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t strideH,
  int32_t strideW, bool pad_valid) {
  int N=1;
  vector<vector<vector<vector<uint64_t>>>> inputArr(N);
  vector<vector<vector<vector<uint64_t>>>> inputMacArr(N);

  vector<vector<vector<vector<uint64_t>>>> filterArr(FH);
  PRG prg;

  for (int i = 0; i < N; i++) {
    inputArr[i].resize(H);
    for (int j = 0; j < H; j++) {
      inputArr[i][j].resize(W);
      for (int k = 0; k < W; k++) {
        inputArr[i][j][k].resize(CI);
        random_mod_p(prg, inputArr[i][j][k].data(), CI, prime_mod);
      }
    }
  }

  for (int i = 0; i < N; i++) {
    inputMacArr[i].resize(H);
    for (int j = 0; j < H; j++) {
      inputMacArr[i][j].resize(W);
      for (int k = 0; k < W; k++) {
        inputMacArr[i][j][k].resize(CI);
        random_mod_p(prg, inputMacArr[i][j][k].data(), CI, prime_mod);
      }
    }
  }

  if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
      filterArr[i].resize(FW);
      for (int j = 0; j < FW; j++) {
        filterArr[i][j].resize(CI);
        for (int k = 0; k < CI; k++) {
          filterArr[i][j][k].resize(CO);
          random_mod_p(prg, filterArr[i][j][k].data(), CO, prime_mod);
        }
      }
    }
  }
  auto start = clock_start();
  he_conv.convolution_gen(H, W, CI, FH, FW, CO, strideH, strideW, pad_valid, inputArr, inputMacArr, filterArr,
                      mod, false, true);
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

int main(int argc, char **argv) {
  parse_arguments(argc, argv, &party, &port);

  cout << "=================================================================="
       << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Mod: " << prime_mod << " - Image: " << image_h << "x" << image_h
       << "x" << inp_chans << " - Filter: " << filter_h << "x" << filter_h
       << "x" << out_chans << "\n- Stride: " << stride << "x" << stride
       << " - Padding: " << pad_l << "x" << pad_r
       << " - # Threads: " << num_threads << endl;
  cout << "=================================================================="
       << endl;

  bool pad_valid;

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);
  uint64_t comm_sent = 0;
  uint64_t start_comm = io->counter;
  auto start = clock_start();
  ConvField he_conv(party, io);
  start_comm = io->counter;
  long long t = time_from(start);

  total_time += t;
  if(choice_nn==MINIONN) {
    image_h = 28;
    image_w = 28;
    filter_h = 5;
    filter_w = 5;
    stride_h = 1;
    stride_w = 1;
    inp_chans = 1;
    out_chans = 16;
    pad_valid = false;
    Conv_First(he_conv, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, stride_h, stride_w, pad_valid);

    image_h = 12;
    image_w = 12;
    filter_h = 5;
    filter_w = 5;
    stride_h = 1;
    stride_w = 1;
    inp_chans = 16;
    out_chans = 16;
    pad_valid = false;
    Conv_Gen(he_conv, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, stride_h, stride_w, pad_valid);
  } else {
    //Layer 1
    image_h = 32;
    image_w = 32;
    filter_h = 3;
    filter_w = 3;
    stride_h = 1;
    stride_w = 1;
    inp_chans = 3;
    out_chans = 64;
    pad_l = 1;
    pad_r = 1;
    pad_valid = true;
    Conv_First(he_conv, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, stride_h, stride_w, pad_valid);

    image_h = 1;
    image_w = 1;
    filter_h = 3;
    filter_w = 3;
    stride_h = 1;
    stride_w = 1;
    inp_chans = 1;
    out_chans = 64;
    pad_l = 1;
    pad_r = 1;
    pad_valid = true;
    Conv_Gen(he_conv, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, stride_h, stride_w, pad_valid);

  }

  //Conv_First(he_conv, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, stride_h, stride_w, true);
  //Conv_Gen(he_conv, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, stride_h, stride_w, true);
  cout << "######################Performance#######################" <<endl;
  cout<<"Time Taken: "<<total_time<<" mus"<<endl;
  //Calculate Communication
  comm_sent = (io->counter-start_comm)>>20;
  cout<<"Sent Data (MB): "<<comm_sent<<endl;
  cout << "########################################################" <<endl;

  io->flush();
  return 0;
}
