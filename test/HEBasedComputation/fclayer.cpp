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

#include "fclayer.h"

using namespace std;
using namespace seal;

/* Helper function for rounding to the next power of 2
 * Credit: https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2 */
inline int next_pow2(int val) {
    return pow(2, ceil(log(val)/log(2)));
}

Ciphertext preprocess_vec(const uint64_t *input, const FCMetadata &data,
                          Encryptor &encryptor, BatchEncoder &batch_encoder) {
  // Create copies of the input vector to fill the ciphertext appropiately.
  // Pack using powers of two for easy rotations later
  vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
  uint64_t size_pow2 = next_pow2(data.image_size);
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
