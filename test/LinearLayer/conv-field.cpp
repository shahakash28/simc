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

#include "conv-field.h"

using namespace std;
using namespace emp;
using namespace seal;
using namespace Eigen;

int rot_count = 0;
int multiplications = 0;
int additions = 0;

Image pad_image(ConvMetadata data, Image &image) {
  int image_h = data.image_h;
  int image_w = data.image_w;
  Image p_image;

  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;
  int pad_top = data.pad_t;
  int pad_left = data.pad_l;

  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (Channel &channel : image) {
    Channel p_channel = Channel::Zero(image_h + pad_h, image_w + pad_w);
    p_channel.block(pad_top, pad_left, image_h, image_w) = channel;
    p_image.push_back(p_channel);
  }
  return p_image;
}

/* Adapted im2col algorithm from Caffe framework */
void i2c(Image &image, Channel &column, const int filter_h, const int filter_w,
         const int stride_h, const int stride_w, const int output_h,
         const int output_w) {
  int height = image[0].rows();
  int width = image[0].cols();
  int channels = image.size();

  int col_width = column.cols();

  // Index counters for images
  int column_i = 0;
  const int channel_size = height * width;
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (auto &channel : image) {
    for (int filter_row = 0; filter_row < filter_h; filter_row++) {
      for (int filter_col = 0; filter_col < filter_w; filter_col++) {
        int input_row = filter_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!condition_check(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              int row_i = column_i / col_width;
              int col_i = column_i % col_width;
              column(row_i, col_i) = 0;
              column_i++;
            }
          } else {
            int input_col = filter_col;
            for (int output_col = output_w; output_col; output_col--) {
              if (condition_check(input_col, width)) {
                int row_i = column_i / col_width;
                int col_i = column_i % col_width;
                column(row_i, col_i) = channel(input_row, input_col);
                column_i++;
              } else {
                int row_i = column_i / col_width;
                int col_i = column_i % col_width;
                column(row_i, col_i) = 0;
                column_i++;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/* Generates a masking vector of random noise that will be applied to parts of the ciphertext
 * that contain leakage from the convolution */
vector<Plaintext> HE_preprocess_noise_plain(const uint64_t* const* secret_share, const ConvMetadata &data,
        BatchEncoder &batch_encoder) {
    vector<vector<uint64_t>> noise(data.out_ct,
                                   vector<uint64_t>(data.slot_count, 0ULL));
    // Sample randomness into vector
    PRG prg;
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
      random_mod_p(prg, noise[ct_idx].data(), data.slot_count,
                                 prime_mod);
    }

    // Puncture the vector with secret share where an actual convolution result value lives
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        int ct_idx = out_c / (2*data.chans_per_half);
        int half_idx = (out_c % (2*data.chans_per_half)) / data.chans_per_half;
        int half_off = out_c % data.chans_per_half;
        for (int col = 0; col < data.output_h; col++) {
            for (int row = 0; row < data.output_w; row++) {
                int noise_idx = half_idx * data.pack_num
                                + half_off * data.image_size
                                + col * data.stride_w * data.image_w
                                + row * data.stride_h;
                int share_idx = col * data.output_w + row ;
                noise[ct_idx][noise_idx] = secret_share[out_c][share_idx];
            }
        }
    }

    // Encrypt all the noise vectors
    vector<Plaintext> enc_noise(data.out_ct);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        batch_encoder.encode(noise[ct_idx], enc_noise[ct_idx]);
    }
    return enc_noise;
}

// Generates a masking vector of random noise that will be applied to parts of
// the ciphertext that contain leakage from the convolution
vector<Ciphertext> HE_preprocess_noise(const uint64_t *const *secret_share,
                                       const ConvMetadata &data,
                                       Encryptor &encryptor,
                                       BatchEncoder &batch_encoder,
                                       Evaluator &evaluator) {
  vector<vector<uint64_t>> noise(data.out_ct,
                                 vector<uint64_t>(data.slot_count, 0ULL));
  // Sample randomness into vector
  PRG prg;
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    random_mod_p(prg, noise[ct_idx].data(), data.slot_count,
                               prime_mod);
  }
  vector<Ciphertext> enc_noise(data.out_ct);

  // Puncture the vector with 0s where an actual convolution result value lives
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    int out_base = 2 * ct_idx * data.chans_per_half;
    for (int out_c = 0;
         out_c < 2 * data.chans_per_half && out_c + out_base < data.out_chans;
         out_c++) {
      int half_idx = out_c / data.chans_per_half;
      int half_off = out_c % data.chans_per_half;
      for (int col = 0; col < data.output_h; col++) {
        for (int row = 0; row < data.output_w; row++) {
          int noise_idx =
              half_idx * data.pack_num + half_off * data.image_size +
              col * data.stride_w * data.image_w + row * data.stride_h;
          int share_idx = col * data.output_w + row;
          noise[ct_idx][noise_idx] = secret_share[out_base + out_c][share_idx];
        }
      }
    }
    Plaintext tmp;
    batch_encoder.encode(noise[ct_idx], tmp);
    encryptor.encrypt(tmp, enc_noise[ct_idx]);
    evaluator.mod_switch_to_next_inplace(enc_noise[ct_idx]);
  }
  return enc_noise;
}

// Preprocesses the input image for output packing. Ciphertext is packed in
// RowMajor order. In this mode simply pack all the input channels as tightly as
// possible where each channel is padded to the nearest of two
vector<vector<uint64_t>> preprocess_image_OP(Image &image, ConvMetadata data) {
  vector<vector<uint64_t>> ct(data.inp_ct,
                              vector<uint64_t>(data.slot_count, 0));
  int inp_c = 0;
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
    int inp_c_limit = (ct_idx + 1) * 2 * data.chans_per_half;
    for (; inp_c < data.inp_chans && inp_c < inp_c_limit; inp_c++) {
      // Calculate which half of ciphertext the output channel
      // falls in and the offest from that half,
      int half_idx = (inp_c % (2 * data.chans_per_half)) / data.chans_per_half;
      int half_off = inp_c % data.chans_per_half;
      for (int row = 0; row < data.image_h; row++) {
        for (int col = 0; col < data.image_w; col++) {
          int idx = half_idx * data.pack_num + half_off * data.image_size +
                    row * data.image_w + col;
          ct[ct_idx][idx] = image[inp_c](row, col);
        }
      }
    }
  }
  return ct;
}

vector<uint64_t> pt_rotate(int slot_count, int rotation, vector<uint64_t> &vec) {
    vector<uint64_t> new_vec(slot_count, 0);
    int pack_num = slot_count / 2;
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int half = 0; half < 2; half++) {
        for (int idx = 0; idx < pack_num; idx++) {
            // Wrap around the half if we accidently pull too far
            int offset = neg_mod(rotation+idx, pack_num);
            new_vec[half*pack_num + idx] = vec[half*pack_num + offset];
        }
    }
    return new_vec;
}

//Additonal Method
template <class T>
vector<T> filter_rotations_dash(T &input,
                                const ConvMetadata &data,
                                Evaluator *evaluator,
                                GaloisKeys *gal_keys) {
  vector<T> rotations(input.size(), T(data.filter_size));
  cout<<"Input Size: "<<input.size()<<endl;
  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;

  // This tells us how many filters fit on a single row of the padded image
  int f_per_row = data.image_w + pad_w - data.filter_w + 1;

  // This offset calculates rotations needed to bring filter from top left
  // corner of image to the top left corner of padded image
  int offset = f_per_row * data.pad_t + data.pad_l;

  // For each element of the filter, rotate the padded image s.t. the top
  // left position always contains the first element of the image it touches
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int f_row = 0; f_row < data.filter_h; f_row++) {
      int row_offset = f_row * data.image_w - offset;
      for (int f_col = 0; f_col < data.filter_w; f_col++) {
          int rot_amt = row_offset + f_col;
          for (int ct_idx = 0; ct_idx < input.size(); ct_idx++) {
              int idx = f_row*data.filter_w+f_col;
              // The constexpr is necessary so the generic type to be used
              // in branch
              if constexpr (is_same<T, vector<vector<uint64_t>>>::value) {
                  rotations[ct_idx][idx] = pt_rotate(data.slot_count,
                                                     rot_amt,
                                                     input[ct_idx]);
              } else {
                  evaluator->rotate_rows(input[ct_idx],
                                        rot_amt,
                                        *gal_keys,
                                        rotations[ct_idx][idx]);
                  rot_count += 1;
              }
          }
      }
  }
  cout<<"Here"<<endl;
  return rotations;
}

// Evaluates the filter rotations necessary to convole an input. Essentially,
// think about placing the filter in the top left corner of the padded image and
// sliding the image over the filter in such a way that we capture which
// elements of filter multiply with which elements of the image. We account for
// the zero padding by zero-puncturing the masks. This function can evaluate
// plaintexts and ciphertexts.
vector<vector<Ciphertext>> filter_rotations(vector<Ciphertext> &input,
                                            const ConvMetadata &data,
                                            Evaluator *evaluator,
                                            GaloisKeys *gal_keys) {
  vector<vector<Ciphertext>> rotations(input.size(),
                                       vector<Ciphertext>(data.filter_size));
  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;

  // This tells us how many filters fit on a single row of the padded image
  int f_per_row = data.image_w + pad_w - data.filter_w + 1;

  // This offset calculates rotations needed to bring filter from top left
  // corner of image to the top left corner of padded image
  int offset = f_per_row * data.pad_t + data.pad_l;

  // For each element of the filter, rotate the padded image s.t. the top
  // left position always contains the first element of the image it touches
#pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for (int f = 0; f < data.filter_size; f++) {
    for (size_t ct_idx = 0; ct_idx < input.size(); ct_idx++) {
      int f_row = f / data.filter_w;
      int f_col = f % data.filter_w;
      int row_offset = f_row * data.image_w - offset;
      int rot_amt = row_offset + f_col;
      int idx = f_row * data.filter_w + f_col;
      evaluator->rotate_rows(input[ct_idx], rot_amt, *gal_keys,
                             rotations[ct_idx][idx]);
    }
  }
  return rotations;
}

// Encrypts the given input image
vector<Ciphertext> HE_encrypt(vector<vector<uint64_t>> &pt,
                              const ConvMetadata &data, Encryptor &encryptor,
                              BatchEncoder &batch_encoder) {
  vector<Ciphertext> ct(pt.size());
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (size_t ct_idx = 0; ct_idx < pt.size(); ct_idx++) {
    Plaintext tmp;
    batch_encoder.encode(pt[ct_idx], tmp);
    encryptor.encrypt(tmp, ct[ct_idx]);
  }
  return ct;
}

/* Encrypts the given input image and all of its rotations */
vector<vector<Ciphertext>> HE_encrypt_rotations(vector<vector<vector<uint64_t>>> &rotations,
        const ConvMetadata &data, Encryptor &encryptor, BatchEncoder &batch_encoder) {
    cout<<"Rotations Size "<<rotations.size()<<", Rotations Zero Size: "<<rotations[0].size()<<endl;
    vector<vector<Ciphertext>> enc_rots(rotations.size(),
                                        vector<Ciphertext>(rotations[0].size()));
    for (int ct_idx = 0; ct_idx < rotations.size(); ct_idx++) {
        for (int f = 0; f < rotations[0].size(); f++) {
            Plaintext tmp;
            batch_encoder.encode(rotations[ct_idx][f], tmp);
            encryptor.encrypt(tmp, enc_rots[ct_idx][f]);
        }
    }
    return enc_rots;
}

vector<vector<vector<Plaintext>>> HE_preprocess_filters(const uint64_t* const* const* filters,
        const ConvMetadata &data, BatchEncoder &batch_encoder) {
    // Mask is convolutions x cts per convolution x mask size
    vector<vector<vector<vector<uint64_t>>>> masks(
            data.convs,
            vector<vector<vector<uint64_t>>>(
                data.inp_ct,
                vector<vector<uint64_t>>(data.filter_size, vector<uint64_t>(data.slot_count, 0))));
    // Since a half in a permutation may have a variable number of rotations we
    // use this index to track where we are at in the masks tensor
    int conv_idx = 0;
    // Build each half permutation as well as it's inward rotations

    for (int perm = 0; perm < data.half_perms; perm += 2) {
        // We populate two different half permutations at a time (if we have at
        // least 2). The second permutation is what you'd get by flipping the
        // columns of the first
        for (int half = 0; half < data.inp_halves; half++) {
            int ct_idx = half / 2;
            int half_idx = half % 2;
            int inp_base = half * data.chans_per_half;

            // The output channel the current ct starts from
            int out_base = (((perm/2) + ct_idx)*2*data.chans_per_half) % data.out_mod;
            // If we're on the last output half, the first and last halves aren't
            // in the same ciphertext, and the last half has repeats, then do
            // repeated packing and skip the second half
            bool last_out = ((out_base + data.out_in_last) == data.out_chans)
                            && data.out_halves != 2;
            bool half_repeats = last_out && data.last_repeats;
            // If the half is repeating we do possibly less number of rotations
            int total_rots = (half_repeats) ? data.last_rots : data.half_rots;
            // Generate all inward rotations of each half
            for (int rot = 0; rot < total_rots; rot++) {
                for (int chan = 0; chan < data.chans_per_half
                                   && (chan + inp_base) < data.inp_chans; chan++) {
                    for (int f = 0; f < data.filter_size; f++) {
                        // Pull the value of this mask
                        int f_w = f % data.filter_w;
                        int f_h = f / data.filter_w;
                        // Set the coefficients of this channel for both
                        // permutations
                        uint64_t val, val2;
                        int out_idx, out_idx2;

                        // If this is a repeating half we first pad out_chans to
                        // nearest power of 2 before repeating
                        if (half_repeats) {
                            out_idx = neg_mod(chan-rot, data.repeat_chans) + out_base;
                            // If we're on a padding channel then val should be 0
                            val = (out_idx < data.out_chans)
                                ? filters[out_idx][inp_base + chan][f] : 0;
                            // Second value will always be 0 since the second
                            // half is empty if we are repeating
                            val2 = 0;
                        } else {
                            int offset = neg_mod(chan-rot, data.chans_per_half);
                            if (half_idx) {
                                // If out_halves < 1 we may repeat within a
                                // ciphertext
                                // TODO: Add the log optimization for this case
                                if (data.out_halves > 1)
                                    out_idx = offset + out_base + data.chans_per_half;
                                else
                                    out_idx = offset + out_base;
                                out_idx2 = offset + out_base;
                            } else {
                                out_idx = offset + out_base;
                                out_idx2 = offset + out_base + data.chans_per_half;
                            }
                            val = (out_idx < data.out_chans)
                                ? filters[out_idx][inp_base+chan][f] : 0;
                            val2 = (out_idx2 < data.out_chans)
                                ? filters[out_idx2][inp_base+chan][f] : 0;
                        }
                        // Iterate through the whole image and figure out which
                        // values the filter value touches - this is the same
                        // as for input packing
                        for(int curr_h = 0; curr_h < data.image_h; curr_h += data.stride_h) {
                            for(int curr_w = 0; curr_w < data.image_w; curr_w += data.stride_w) {
                                // curr_h and curr_w simulate the current top-left position of
                                // the filter. This detects whether the filter would fit over
                                // this section. If it's out-of-bounds we set the mask index to 0
                                bool zero = ((curr_w+f_w) < data.pad_l) ||
                                    ((curr_w+f_w) >= (data.image_w+data.pad_l)) ||
                                    ((curr_h+f_h) < data.pad_t) ||
                                    ((curr_h+f_h) >= (data.image_h+data.pad_l));
                                // Calculate which half of ciphertext the output channel
                                // falls in and the offest from that half,
                                int idx = half_idx * data.pack_num
                                        + chan * data.image_size
                                        + curr_h * data.image_h
                                        + curr_w;
                                // Add both values to appropiate permutations
                                masks[conv_idx+rot][ct_idx][f][idx] = zero? 0: val;
                                if (data.half_perms > 1) {
                                    masks[conv_idx+data.half_rots+rot][ct_idx][f][idx] = zero? 0: val2;
                                }
                            }
                        }
                    }
                }
            }
        }
        conv_idx += 2*data.half_rots;
    }

    // Encode all the masks
    vector<vector<vector<Plaintext>>> encoded_masks(
            data.convs,
            vector<vector<Plaintext>>(
                data.inp_ct,
                vector<Plaintext>(data.filter_size)));
    for (int conv = 0; conv < data.convs; conv++) {
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
            for (int f = 0; f < data.filter_size; f++) {
                batch_encoder.encode(masks[conv][ct_idx][f],
                                         encoded_masks[conv][ct_idx][f]);
            }
        }
    }
    return encoded_masks;
}

// Creates filter masks for an image input that has been output packed.
vector<vector<vector<Plaintext>>>
HE_preprocess_filters_OP(Filters &filters, const ConvMetadata &data,
                         BatchEncoder &batch_encoder) {
  // Mask is convolutions x cts per convolution x mask size
  vector<vector<vector<Plaintext>>> encoded_masks(
      data.convs, vector<vector<Plaintext>>(
                      data.inp_ct, vector<Plaintext>(data.filter_size)));
  // Since a half in a permutation may have a variable number of rotations we
  // use this index to track where we are at in the masks tensor
  // Build each half permutation as well as it's inward rotations
#pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for (int perm = 0; perm < data.half_perms; perm += 2) {
    for (int rot = 0; rot < data.half_rots; rot++) {
      int conv_idx = perm * data.half_rots;
      for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        // The output channel the current ct starts from
        // int out_base = (((perm/2) + ct_idx)*2*data.chans_per_half) %
        // data.out_mod;
        int out_base = (perm * data.chans_per_half) % data.out_mod;
        // Generate all inward rotations of each half -- half_rots loop
        for (int f = 0; f < data.filter_size; f++) {
          vector<vector<uint64_t>> masks(2,
                                         vector<uint64_t>(data.slot_count, 0));
          for (int half_idx = 0; half_idx < 2; half_idx++) {
            int inp_base = (2 * ct_idx + half_idx) * data.chans_per_half;
            for (int chan = 0; chan < data.chans_per_half &&
                               (chan + inp_base) < data.inp_chans;
                 chan++) {
              // Pull the value of this mask
              int f_w = f % data.filter_w;
              int f_h = f / data.filter_w;
              // Set the coefficients of this channel for both
              // permutations
              uint64_t val, val2;
              int out_idx, out_idx2;

              int offset = neg_mod(chan - rot, (int64_t)data.chans_per_half);
              if (half_idx) {
                // If out_halves < 1 we may repeat within a
                // ciphertext
                // TODO: Add the log optimization for this case
                if (data.out_halves > 1)
                  out_idx = offset + out_base + data.chans_per_half;
                else
                  out_idx = offset + out_base;
                out_idx2 = offset + out_base;
              } else {
                out_idx = offset + out_base;
                out_idx2 = offset + out_base + data.chans_per_half;
              }
              val = (out_idx < data.out_chans)
                        ? filters[out_idx][inp_base + chan](f_h, f_w)
                        : 0;
              val2 = (out_idx2 < data.out_chans)
                         ? filters[out_idx2][inp_base + chan](f_h, f_w)
                         : 0;
              // Iterate through the whole image and figure out which
              // values the filter value touches - this is the same
              // as for input packing
              for (int curr_h = 0; curr_h < data.image_h;
                   curr_h += data.stride_h) {
                for (int curr_w = 0; curr_w < data.image_w;
                     curr_w += data.stride_w) {
                  // curr_h and curr_w simulate the current top-left position of
                  // the filter. This detects whether the filter would fit over
                  // this section. If it's out-of-bounds we set the mask index
                  // to 0
                  bool zero = ((curr_w + f_w) < data.pad_l) ||
                              ((curr_w + f_w) >= (data.image_w + data.pad_l)) ||
                              ((curr_h + f_h) < data.pad_t) ||
                              ((curr_h + f_h) >= (data.image_h + data.pad_l));
                  // Calculate which half of ciphertext the output channel
                  // falls in and the offest from that half,
                  int idx = half_idx * data.pack_num + chan * data.image_size +
                            curr_h * data.image_w + curr_w;
                  // Add both values to appropiate permutations
                  masks[0][idx] = zero ? 0 : val;
                  if (data.half_perms > 1) {
                    masks[1][idx] = zero ? 0 : val2;
                  }
                }
              }
            }
          }
          batch_encoder.encode(masks[0],
                               encoded_masks[conv_idx + rot][ct_idx][f]);
          if (data.half_perms > 1) {
            batch_encoder.encode(
                masks[1],
                encoded_masks[conv_idx + data.half_rots + rot][ct_idx][f]);
          }
        }
      }
    }
  }
  return encoded_masks;
}

vector<vector<Ciphertext>> HE_conv(vector<vector<vector<Plaintext>>> &masks,
        vector<vector<Ciphertext>> &rotations, const ConvMetadata &data, Evaluator &evaluator,
        RelinKeys &relin_keys, Ciphertext &zero) {
    vector<vector<Ciphertext>> result(data.convs, vector<Ciphertext>(data.inp_ct));
    // Init the result vector to all 0
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int conv = 0; conv < data.convs; conv++) {
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++)
            result[conv][ct_idx] = zero;
    }
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    // Multiply masks and add for each convolution
    for (int perm = 0; perm < data.half_perms; perm++) {
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
            // The output channel the current ct starts from
            int out_base = ((perm/2+ct_idx)*2*data.chans_per_half) % data.out_mod;
            // If we're on the last output half, the first and last halves aren't
            // in the same ciphertext, and the last half has repeats, then only
            // convolve last_rots number of times
            bool last_out = ((out_base + data.out_in_last) == data.out_chans)
                            && data.out_halves != 2;
            bool half_repeats = last_out && data.last_repeats;
            int total_rots = (half_repeats) ? data.last_rots : data.half_rots;
            for (int rot = 0; rot < total_rots; rot++) {
                for (int f = 0; f < data.filter_size; f++) {
                    // Note that if a mask is zero this will result in a
                    // 'transparent' ciphertext which SEAL won't allow by default.
                    // This isn't a problem however since we're adding the result
                    // with something else, and the size is known beforehand so
                    // having some elements be 0 doesn't matter
                    Ciphertext tmp;
                    evaluator.multiply_plain(rotations[ct_idx][f],
                                                 masks[perm*data.half_rots+rot][ct_idx][f],
                                                 tmp);
                    evaluator.relinearize_inplace(tmp, relin_keys);
                    multiplications += 1;
                    evaluator.add_inplace(result[perm*data.half_rots+rot][ct_idx], tmp);
                    additions += 1;
                }
            }
        }
    }
    return result;
}

// Performs convolution for an output packed image. Returns the intermediate
// rotation sets
vector<Ciphertext> HE_conv_OP(vector<vector<vector<Plaintext>>> &masks,
                              vector<vector<Ciphertext>> &rotations,
                              const ConvMetadata &data, Evaluator &evaluator,
                              Ciphertext &zero) {
  vector<Ciphertext> result(data.convs);

  // Multiply masks and add for each convolution
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int conv_idx = 0; conv_idx < data.convs; conv_idx++) {
    result[conv_idx] = zero;
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
      for (int f = 0; f < data.filter_size; f++) {
        Ciphertext tmp;
        if (!masks[conv_idx][ct_idx][f].is_zero()) {
          evaluator.multiply_plain(rotations[ct_idx][f],
                                   masks[conv_idx][ct_idx][f], tmp);

          evaluator.add_inplace(result[conv_idx], tmp);
        }
      }
    }
    evaluator.mod_switch_to_next_inplace(result[conv_idx]);
  }
  return result;
}

// Takes the result of an output-packed convolution, and rotates + adds all the
// ciphertexts to get a tightly packed output
vector<Ciphertext> HE_output_rotations(vector<Ciphertext> &convs,
                                       const ConvMetadata &data,
                                       Evaluator &evaluator,
                                       GaloisKeys &gal_keys, Ciphertext &zero,
                                       vector<Ciphertext> &enc_noise) {
  vector<Ciphertext> partials(data.half_perms);
  Ciphertext zero_next_level = zero;
  evaluator.mod_switch_to_next_inplace(zero_next_level);
  // Init the result vector to all 0
  vector<Ciphertext> result(data.out_ct);
  #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    result[ct_idx] = zero_next_level;
  }

  // For each half perm, add up all the inside channels of each half
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int perm = 0; perm < data.half_perms; perm += 2) {
    partials[perm] = zero_next_level;
    if (data.half_perms > 1)
      partials[perm + 1] = zero_next_level;
    // The output channel the current ct starts from
    int total_rots = data.half_rots;
    for (int in_rot = 0; in_rot < total_rots; in_rot++) {
      int conv_idx = perm * data.half_rots + in_rot;
      int rot_amt;
      rot_amt =
          -neg_mod(-in_rot, (int64_t)data.chans_per_half) * data.image_size;

      evaluator.rotate_rows_inplace(convs[conv_idx], rot_amt, gal_keys);
      evaluator.add_inplace(partials[perm], convs[conv_idx]);
      // Do the same for the column swap if it exists
      if (data.half_perms > 1) {
        evaluator.rotate_rows_inplace(convs[conv_idx + data.half_rots], rot_amt,
                                      gal_keys);
        evaluator.add_inplace(partials[perm + 1],
                              convs[conv_idx + data.half_rots]);
      }
    }
    // The correct index for the correct ciphertext in the final output
    int out_idx = (perm / 2) % data.out_ct;
    if (perm == 0) {
      // The first set of convolutions is aligned correctly
      evaluator.add_inplace(result[out_idx], partials[perm]);
      ///*
      if (data.out_halves == 1 && data.inp_halves > 1) {
        // If the output fits in a single half but the input
        // doesn't, add the two columns
        evaluator.rotate_columns_inplace(partials[perm], gal_keys);
        evaluator.add_inplace(result[out_idx], partials[perm]);
      }
      //*/
      // Do the same for column swap if exists and we aren't on a repeat
      if (data.half_perms > 1) {
        evaluator.rotate_columns_inplace(partials[perm + 1], gal_keys);
        evaluator.add_inplace(result[out_idx], partials[perm + 1]);
      }
    } else {
      // Rotate the output ciphertexts by one and add
      evaluator.add_inplace(result[out_idx], partials[perm]);
      // If we're on a tight half we add both halves together and
      // don't look at the column flip
      if (data.half_perms > 1) {
        evaluator.rotate_columns_inplace(partials[perm + 1], gal_keys);
        evaluator.add_inplace(result[out_idx], partials[perm + 1]);
      }
    }
  }
  //// Add the noise vector to remove any leakage
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    evaluator.add_inplace(result[ct_idx], enc_noise[ct_idx]);
    // evaluator.mod_switch_to_next_inplace(result[ct_idx]);
  }
  return result;
}

vector<Ciphertext> HE_output_rotations_dash(vector<vector<Ciphertext>> convs,
        const ConvMetadata &data, Evaluator &evaluator, GaloisKeys &gal_keys,
        Ciphertext &zero) {
    vector<vector<Ciphertext>> partials(data.half_perms,
                                        vector<Ciphertext>(data.inp_ct));
    // Init the result vector to all 0
    vector<Ciphertext> result(data.out_ct);
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        result[ct_idx] = zero;
    }
    // For each half perm, add up all the inside channels of each half
    for (int perm = 0; perm < data.half_perms; perm+=2) {
        int rot;
        // Can save an addition or so by initially setting the partials vector
        // to a convolution result if it's correctly aligned. Otherwise init to
        // all 0s
        if (data.inp_chans <= data.out_chans || data.out_chans == 1) {
            for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
                partials[perm][ct_idx] = convs[perm*data.half_rots][ct_idx];
                if (data.half_perms > 1)
                    partials[perm+1][ct_idx] = convs[(perm+1)*data.half_rots][ct_idx];;
            }
            rot = 1;
        } else {
            for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
                partials[perm][ct_idx] = zero;
                if (data.half_perms > 1)
                    partials[perm+1][ct_idx] = zero;
            }
            rot = 0;
        }
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
            // The output channel the current ct starts from
            int out_base = ((perm/2+ct_idx)*2*data.chans_per_half) % data.out_mod;
            // Whether we are on the last input half
            bool last_in = (perm + ct_idx + 1) % (data.inp_ct) == 0;
            // If we're on the last output half, the first and last halves aren't
            // in the same ciphertext, and the last half has repeats, then do the
            // rotations optimization when summing up
            bool last_out = ((out_base + data.out_in_last) == data.out_chans)
                            && data.out_halves != 2;
            bool half_repeats = last_out && data.last_repeats;
            int total_rots = (half_repeats) ? data.last_rots : data.half_rots;
            for (int in_rot = rot; in_rot < total_rots; in_rot++) {
                int conv_idx = perm * data.half_rots + in_rot;
                int rot_amt;
                // If we're on a repeating half the amount we rotate will be
                // different
                if (half_repeats)
                    rot_amt = -neg_mod(-in_rot, data.repeat_chans) * data.image_size;
                else
                    rot_amt = -neg_mod(-in_rot, data.chans_per_half) * data.image_size;

                evaluator.rotate_rows_inplace(convs[conv_idx][ct_idx],
                                                  rot_amt,
                                                  gal_keys);
                evaluator.add_inplace(partials[perm][ct_idx],
                                          convs[conv_idx][ct_idx]);
                // Do the same for the column swap if it exists
                if (data.half_perms > 1) {
                    evaluator.rotate_rows_inplace(convs[conv_idx+data.half_rots][ct_idx],
                                                      rot_amt,
                                                      gal_keys);
                    evaluator.add_inplace(partials[perm+1][ct_idx],
                                              convs[conv_idx+data.half_rots][ct_idx]);
                    if (rot_amt != 0)
                        rot_count += 1;
                    additions += 1;
                }
                if (rot_amt != 0)
                    rot_count += 1;
                additions += 1;
            }
            // Add up a repeating half
            if (half_repeats) {
                // If we're on the last inp_half then we might be able to do
                // less rotations. We may be able to find a power of 2 less
                // than chans_per_half that contains all of our needed repeats
                int size_to_reduce;
                if (last_in) {
                    int num_repeats = ceil((float) data.inp_in_last / data.repeat_chans);
                    //  We round the repeats to the closest power of 2
                    int effective_repeats;
                    // When we rotated in the previous loop we cause a bit of overflow
                    // (one extra repeat_chans worth). If the extra overflow fits
                    // into the modulo of the last repeat_chan we can do one
                    // less rotation
                    if (data.repeat_chans*num_repeats % data.inp_in_last == data.repeat_chans - 1)
                        effective_repeats = pow(2, ceil(log(num_repeats)/log(2)));
                    else
                        effective_repeats = pow(2, ceil(log(num_repeats+1)/log(2)));
                    // If the overflow ended up wrapping around then we simply
                    // want chans_per_half as our size
                    size_to_reduce = min(effective_repeats*data.repeat_chans, data.chans_per_half);
                } else
                    size_to_reduce = data.chans_per_half;
                // Perform the actual rotations
                for (int in_rot = size_to_reduce/2; in_rot >= data.repeat_chans; in_rot = in_rot/2) {
                    int rot_amt = in_rot * data.image_size;
                    Ciphertext tmp = partials[perm][ct_idx];
                    evaluator.rotate_rows_inplace(tmp, rot_amt, gal_keys);
                    evaluator.add_inplace(partials[perm][ct_idx], tmp);
                    // Do the same for column swap if exists
                    if (data.half_perms > 1) {
                        tmp = partials[perm+1][ct_idx];
                        evaluator.rotate_rows_inplace(tmp, rot_amt, gal_keys);
                        evaluator.add_inplace(partials[perm+1][ct_idx], tmp);
                        if (rot_amt != 0)
                            rot_count += 1;
                        additions += 1;
                    }
                    if (rot_amt != 0)
                        rot_count += 1;
                    additions += 1;
                }
            }
            // The correct index for the correct ciphertext in the final output
            int out_idx = (perm/2 + ct_idx) % data.out_ct;
            if (perm == 0) {
                // The first set of convolutions is aligned correctly
                evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                if (data.out_halves == 1 && data.inp_halves > 1) {
                    // If the output fits in a single half but the input
                    // doesn't, add the two columns
                    evaluator.rotate_columns_inplace(partials[perm][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                }
                // Do the same for column swap if exists and we aren't on a repeat
                if (data.half_perms > 1 && !half_repeats) {
                    evaluator.rotate_columns_inplace(partials[perm+1][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm+1][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                }
            } else {
                // Rotate the output ciphertexts by one and add
                evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                additions += 1;
                // If we're on a tight half we add both halves together and
                // don't look at the column flip
                if (half_repeats) {
                    evaluator.rotate_columns_inplace(partials[perm][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                } else if (data.half_perms > 1) {
                    evaluator.rotate_columns_inplace(partials[perm+1][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm+1][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                }
            }
        }
    }
    return result;
}

// Decrypts and reshapes convolution result
uint64_t **HE_decrypt(vector<Ciphertext> &enc_result, const ConvMetadata &data,
                      Decryptor &decryptor, BatchEncoder &batch_encoder) {
  // Decrypt ciphertext
  vector<vector<uint64_t>> result(data.out_ct);

#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    Plaintext tmp;
    decryptor.decrypt(enc_result[ct_idx], tmp);
    batch_encoder.decode(tmp, result[ct_idx]);
  }

  uint64_t **final_result = new uint64_t *[data.out_chans];
  // Extract correct values to reshape
  for (int out_c = 0; out_c < data.out_chans; out_c++) {
    int ct_idx = out_c / (2 * data.chans_per_half);
    int half_idx = (out_c % (2 * data.chans_per_half)) / data.chans_per_half;
    int half_off = out_c % data.chans_per_half;
    // Depending on the padding type and stride the output values won't be
    // lined up so extract them into a temporary channel before placing
    // them in resultant Image
    final_result[out_c] = new uint64_t[data.output_h * data.output_w];
    for (int col = 0; col < data.output_h; col++) {
      for (int row = 0; row < data.output_w; row++) {
        int idx = half_idx * data.pack_num + half_off * data.image_size +
                  col * data.stride_w * data.image_w + row * data.stride_h;
        final_result[out_c][col * data.output_w + row] = result[ct_idx][idx];
      }
    }
  }
  return final_result;
}

ConvField::ConvField(int party, NetIO *io) {
  this->party = party;
  this->io = io;
  this->slot_count = POLY_MOD_DEGREE;
  generate_new_keys(party, io, slot_count, context[1], encryptor[1],
                    decryptor[1], evaluator[1], encoder[1], gal_keys[1], relin_keys[1],
                    zero[1]);
  this->slot_count = POLY_MOD_DEGREE;
  generate_new_keys(party, io, slot_count, context[0], encryptor[0],
                    decryptor[0], evaluator[0], encoder[0], gal_keys[0], relin_keys[0],
                    zero[0]);
  cout<<"CP 7"<<endl;
}

ConvField::~ConvField() {
  for (int i = 0; i < 2; i++) {
    free_keys(party, encryptor[i], decryptor[i], evaluator[i], encoder[i],
              gal_keys[i], relin_keys[i], zero[i]);
  }
}

void ConvField::configure_1() {
  data.slot_count = this->slot_count;
  // If using Output packing we pad image_size to the nearest power of 2
  data.image_size = next_pow2(data.image_h * data.image_w);
  data.filter_size = data.filter_h * data.filter_w;

  data.pack_num = slot_count / 2;
  data.chans_per_half = data.pack_num / data.image_size;
  data.out_ct = ceil((float)data.out_chans / (2 * data.chans_per_half));
  data.inp_ct = ceil((float)data.inp_chans / (2 * data.chans_per_half));

  data.inp_halves = ceil((float)data.inp_chans / data.chans_per_half);
  data.out_halves = ceil((float)data.out_chans / data.chans_per_half);

  data.out_mod = data.out_ct * 2 * data.chans_per_half;

  data.half_perms = (data.out_halves % 2 != 0 && data.out_halves > 1)
                        ? data.out_halves + 1
                        : data.out_halves;
  assert(data.out_chans > 0 && data.input_chans > 0);
  assert(data.image_size < (data.slot_count/2));

  data.out_in_last = (data.out_chans % data.chans_per_half) ?
        (data.out_chans % (2*data.chans_per_half)) % data.chans_per_half : data.chans_per_half;
  data.inp_in_last = (data.inp_chans % data.chans_per_half) ?
        (data.inp_chans % (2*data.chans_per_half)) % data.chans_per_half : data.chans_per_half;

  data.repeat_chans = next_pow2(data.out_in_last);
  data.last_repeats = (data.out_in_last <= data.chans_per_half/2) && (data.out_halves % 2 == 1);

  data.half_rots =
      (data.inp_halves > 1 || data.out_halves > 1)
          ? data.chans_per_half
          : max(data.chans_per_half, max(data.out_chans, data.inp_chans));

 data.last_rots = (data.last_repeats) ? data.repeat_chans : data.half_rots;
 data.convs = (data.out_halves == 1) ? data.last_rots : data.half_perms * data.half_rots;
 if (data.pad_valid) {
        data.output_h = ceil((float)(data.image_h - data.filter_h + 1) / data.stride_h);
        data.output_w = ceil((float)(data.image_w - data.filter_w + 1) / data.stride_w);
        data.pad_t = 0;
        data.pad_b = 0;
        data.pad_r = 0;
        data.pad_l = 0;
    } else {
        data.output_h = ceil((float)data.image_h / data.stride_h);
        data.output_w = ceil((float)data.image_w / data.stride_w);
        // Total amount of vertical and horizontal padding needed
        int pad_h = max((data.output_h - 1) * data.stride_h + data.filter_h - data.image_h, 0);
        int pad_w = max((data.output_w - 1) * data.stride_w + data.filter_w - data.image_w, 0);
        // Individual side padding
        data.pad_t = floor((float)pad_h / 2);
        data.pad_b = pad_h - data.pad_t;
        data.pad_l = floor((float)pad_w / 2);
        data.pad_r = pad_w - data.pad_l;
    }
}

void ConvField::configure() {
  data.slot_count = this->slot_count;
  // If using Output packing we pad image_size to the nearest power of 2
  data.image_size = next_pow2(data.image_h * data.image_w);
  data.filter_size = data.filter_h * data.filter_w;

  assert(data.out_chans > 0 && data.inp_chans > 0);
  // Doesn't currently support a channel being larger than a half ciphertext
  assert(data.image_size <= (slot_count / 2));

  data.pack_num = slot_count / 2;
  data.chans_per_half = data.pack_num / data.image_size;
  data.out_ct = ceil((float)data.out_chans / (2 * data.chans_per_half));
  data.inp_ct = ceil((float)data.inp_chans / (2 * data.chans_per_half));

  data.inp_halves = ceil((float)data.inp_chans / data.chans_per_half);
  data.out_halves = ceil((float)data.out_chans / data.chans_per_half);

  // The modulo is calculated per ciphertext instead of per half since we
  // should never have the last out_half wrap around to the first in the
  // same ciphertext
  data.out_mod = data.out_ct * 2 * data.chans_per_half;

  data.half_perms = (data.out_halves % 2 != 0 && data.out_halves > 1)
                        ? data.out_halves + 1
                        : data.out_halves;
  data.half_rots =
      (data.inp_halves > 1 || data.out_halves > 1)
          ? data.chans_per_half
          : max(data.chans_per_half, max(data.out_chans, data.inp_chans));
  data.convs = data.half_perms * data.half_rots;

  data.output_h = 1 + (data.image_h + data.pad_t + data.pad_b - data.filter_h) /
                          data.stride_h;
  data.output_w = 1 + (data.image_w + data.pad_l + data.pad_r - data.filter_w) /
                          data.stride_w;
}

Image ConvField::ideal_functionality(Image &image, Filters &filters) {
  int channels = data.inp_chans;
  int filter_h = data.filter_h;
  int filter_w = data.filter_w;
  int output_h = data.output_h;
  int output_w = data.output_w;

  auto p_image = pad_image(data, image);
  const int col_height = filter_h * filter_w * channels;
  const int col_width = output_h * output_w;
  Channel image_col(col_height, col_width);
  i2c(p_image, image_col, data.filter_h, data.filter_w, data.stride_h,
      data.stride_w, data.output_h, data.output_w);

  // For each filter, flatten it into and multiply with image_col
  Image result;
  for (auto &filter : filters) {
    Channel filter_col(1, col_height);
    // Use im2col with a filter size 1x1 to translate
    i2c(filter, filter_col, 1, 1, 1, 1, filter_h, filter_w);
    Channel tmp = filter_col * image_col;

    // Reshape result of multiplication to the right size
    // SEAL stores matrices in RowMajor form
    result.push_back(Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>(
        tmp.data(), output_h, output_w));
  }
  return result;
}

void ConvField::non_strided_conv(int32_t H, int32_t W, int32_t CI, int32_t FH,
                                 int32_t FW, int32_t CO, Image *image,
                                 Filters *filters,
                                 vector<vector<vector<uint64_t>>> &outArr,
                                 bool verbose) {
  data.image_h = H;
  data.image_w = W;
  data.inp_chans = CI;
  data.out_chans = CO;
  data.filter_h = FH;
  data.filter_w = FW;
  data.pad_t = 0;
  data.pad_b = 0;
  data.pad_l = 0;
  data.pad_r = 0;
  data.stride_h = 1;
  data.stride_w = 1;
  this->slot_count =
      min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, 2 * next_pow2(H * W)));
  configure();

  shared_ptr<SEALContext> context_;
  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  RelinKeys *relin_keys_;
  Ciphertext *zero_;
  if (slot_count == POLY_MOD_DEGREE) {
    context_ = this->context[0];
    encryptor_ = this->encryptor[0];
    decryptor_ = this->decryptor[0];
    evaluator_ = this->evaluator[0];
    encoder_ = this->encoder[0];
    gal_keys_ = this->gal_keys[0];
    relin_keys_ = this->relin_keys[0];
    zero_ = this->zero[0];
  } else if (slot_count == POLY_MOD_DEGREE_LARGE) {
    context_ = this->context[1];
    encryptor_ = this->encryptor[1];
    decryptor_ = this->decryptor[1];
    evaluator_ = this->evaluator[1];
    encoder_ = this->encoder[1];
    gal_keys_ = this->gal_keys[1];
    relin_keys_ = this->relin_keys[1];
    zero_ = this->zero[1];
  } else {
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_,
                      evaluator_, encoder_, gal_keys_, relin_keys_, zero_, verbose);
  }

  if (party == BOB) {
    auto pt = preprocess_image_OP(*image, data);
    if (verbose)
      cout << "[Client] Image preprocessed" << endl;

    auto ct = HE_encrypt(pt, data, *encryptor_, *encoder_);
    send_encrypted_vector(io, ct);
    if (verbose)
      cout << "[Client] Image encrypted and sent" << endl;

    vector<Ciphertext> enc_result(data.out_ct);
    recv_encrypted_vector(io, context_, enc_result);
    auto HE_result = HE_decrypt(enc_result, data, *decryptor_, *encoder_);

    if (verbose)
      cout << "[Client] Result received and decrypted" << endl;

    for (int idx = 0; idx < data.output_h * data.output_w; idx++) {
      for (int chan = 0; chan < CO; chan++) {
        outArr[idx / data.output_w][idx % data.output_w][chan] +=
            HE_result[chan][idx];
      }
    }
  } else // party == ALICE
  {
    PRG prg;
    uint64_t **secret_share = new uint64_t *[CO];
    for (int chan = 0; chan < CO; chan++) {
      secret_share[chan] = new uint64_t[data.output_h * data.output_w];
      random_mod_p(prg, secret_share[chan],
                                 data.output_h * data.output_w, prime_mod);
    }
    vector<Ciphertext> noise_ct = HE_preprocess_noise(
        secret_share, data, *encryptor_, *encoder_, *evaluator_);

    if (verbose)
      cout << "[Server] Noise processed" << endl;

    vector<vector<vector<Plaintext>>> masks_OP;
    masks_OP = HE_preprocess_filters_OP(*filters, data, *encoder_);

    if (verbose)
      cout << "[Server] Filters processed" << endl;

    vector<Ciphertext> result;
    vector<Ciphertext> ct(data.inp_ct);
    vector<vector<Ciphertext>> rotations(data.inp_ct);
    for (int i = 0; i < data.inp_ct; i++) {
      rotations[i].resize(data.filter_size);
    }
    recv_encrypted_vector(io, context_, ct);
    rotations = filter_rotations(ct, data, evaluator_, gal_keys_);
    if (verbose)
      cout << "[Server] Filter Rotations done" << endl;

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, rotations[0][0],
                       "before homomorphic convolution");
#endif

    auto conv_result =
        HE_conv_OP(masks_OP, rotations, data, *evaluator_, *zero_);
    if (verbose)
      cout << "[Server] Convolution done" << endl;

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, conv_result[0],
                       "after homomorphic convolution");
#endif

    result = HE_output_rotations(conv_result, data, *evaluator_, *gal_keys_,
                                 *zero_, noise_ct);
    if (verbose)
      cout << "[Server] Output Rotations done" << endl;

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, result[0], "after output rotations");
#endif

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data =
        context_->get_context_data(parms_id);
    for (size_t ct_idx = 0; ct_idx < result.size(); ct_idx++) {
      flood_ciphertext(result[ct_idx], context_data, SMUDGING_BITLEN);
    }

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, result[0], "after noise flooding");
#endif

    for (size_t ct_idx = 0; ct_idx < result.size(); ct_idx++) {
      evaluator_->mod_switch_to_next_inplace(result[ct_idx]);
    }

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, result[0], "after mod-switch");
#endif

    send_encrypted_vector(io, result);
    if (verbose)
      cout << "[Server] Result computed and sent" << endl;

    for (int idx = 0; idx < data.output_h * data.output_w; idx++) {
      for (int chan = 0; chan < CO; chan++) {
        outArr[idx / data.output_w][idx % data.output_w][chan] +=
            (prime_mod - secret_share[chan][idx]);
      }
    }
    for (int i = 0; i < data.out_chans; i++)
      delete[] secret_share[i];
    delete[] secret_share;
  }
  if (slot_count > POLY_MOD_DEGREE && slot_count < POLY_MOD_DEGREE_LARGE) {
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, relin_keys_,
              zero_);
  }
}

uint64_t** conv_preprocess_invert(vector<Ciphertext> &r_mac_ct, const ConvMetadata &data, Decryptor &decryptor,
        BatchEncoder &batch_encoder) {
    // Decrypt ciphertext
    vector<vector<uint64_t>> result(data.inp_ct);
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        Plaintext tmp;
        decryptor.decrypt(r_mac_ct[ct_idx], tmp);
        batch_encoder.decode(tmp, result[ct_idx]);
    }
    uint64_t** final_result = new uint64_t*[data.inp_chans];
    int inp_c = 0;
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        int inp_c_limit = (ct_idx+1) * 2 * data.chans_per_half;
        for (;inp_c < data.inp_chans && inp_c < inp_c_limit; inp_c++) {
            final_result[inp_c] = new uint64_t[data.image_size];
            // Calculate which half of ciphertext the output channel
            // falls in and the offest from that half,
            int half_idx = (inp_c % (2*data.chans_per_half)) / data.chans_per_half;
            int half_off = inp_c % data.chans_per_half;
            for (int row = 0; row < data.image_h; row++) {
                for (int col = 0; col < data.image_w; col++) {
                    int idx = half_idx * data.pack_num
                            + half_off * data.image_size
                            + row * data.image_h
                            + col;
                    final_result[inp_c][row*data.image_h + col] = result[ct_idx][idx];
                }
            }
        }
    }
    return final_result;
}

void ConvField::convolution_first(int32_t H, int32_t W, int32_t CI,
                            int32_t FH, int32_t FW, int32_t CO,
                            int32_t strideH, int32_t strideW, bool pad_valid,
                            vector<vector<vector<vector<uint64_t>>>> &inputArr,
                            vector<vector<vector<vector<uint64_t>>>> &filterArr,
                            bool verify_output, bool verbose) {
  data.image_h = H;
  data.image_w = W;
  data.inp_chans = CI;
  data.out_chans = CO;
  data.filter_h = FH;
  data.filter_w = FW;
  data.pad_valid = pad_valid;
  data.stride_h = strideH;
  data.stride_w = strideW;
  this->slot_count =
      min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, 2 * next_pow2(H * W)));
  cout<<"CP 1"<<endl;
  configure_1();

  shared_ptr<SEALContext> context_;
  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  RelinKeys *relin_keys_;
  Ciphertext *zero_;
  if (slot_count == POLY_MOD_DEGREE) {
    context_ = this->context[0];
    encryptor_ = this->encryptor[0];
    decryptor_ = this->decryptor[0];
    evaluator_ = this->evaluator[0];
    encoder_ = this->encoder[0];
    gal_keys_ = this->gal_keys[0];
    relin_keys_ = this->relin_keys[0];
    zero_ = this->zero[0];
  } else if (slot_count == POLY_MOD_DEGREE_LARGE) {
    context_ = this->context[1];
    encryptor_ = this->encryptor[1];
    decryptor_ = this->decryptor[1];
    evaluator_ = this->evaluator[1];
    encoder_ = this->encoder[1];
    gal_keys_ = this->gal_keys[1];
    relin_keys_ = this->relin_keys[1];
    zero_ = this->zero[1];
  } else {
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_,
                      evaluator_, encoder_, gal_keys_, relin_keys_, zero_, verbose);
  }
  cout<<"CP 2"<<endl;

  Image image;
  //Filters filters;

  if(party == BOB) {
    //Cast to Image
    image.resize(CI);
    for (int chan = 0; chan < CI; chan++) {
      Channel tmp_chan(H, W);
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          tmp_chan(h, w) = inputArr[0][h][w][chan];
        }
      }
      image[chan] = tmp_chan;
    }

    //Prepare and send ciphertext
    auto pt = preprocess_image_OP(image, data);
    auto rotated_pt = filter_rotations_dash(pt, data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, data, *encryptor_, *encoder_);
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());
    cout<<"Ciphertext size is: "<<ct_flat_rotations.size()<<endl;
    cout<<"CT Ciphertext size is: "<<ct_rotations[0].size()<<endl;
    send_encrypted_vector(io, ct_flat_rotations);

    //Receive ciphertexts
    vector<Ciphertext> enc_inp_result(data.out_ct);
    vector<Ciphertext> enc_mac_result(data.out_ct);
    recv_encrypted_vector(io, context_, enc_inp_result);
    recv_encrypted_vector(io, context_, enc_mac_result);
    auto input_share_1 = HE_decrypt(enc_inp_result, data, *decryptor_, *encoder_);
    auto input_mac_share_1 = HE_decrypt(enc_mac_result, data, *decryptor_, *encoder_);

  } else {
    //Prepare shares
    PRG prg;
    //Mac key
    uint64_t mac_key;
    random_mod_p(prg, &mac_key, 1, prime_mod);
    vector<uint64_t> mac_vec(encoder_->slot_count(), mac_key);
    Plaintext* enc_mac = new Plaintext();
    encoder_->encode(mac_vec, *enc_mac);

    //Output shares
    uint64_t **input_share = new uint64_t *[data.out_chans];
    for (int chan = 0; chan < data.out_chans; chan++) {
      input_share[chan] = new uint64_t[data.output_h * data.output_w];
      random_mod_p(prg, input_share[chan],
                                 data.output_h * data.output_w, prime_mod);
    }
    vector<Plaintext> linear = HE_preprocess_noise_plain(input_share, data, *encoder_);

    uint64_t **input_share_mac = new uint64_t *[data.out_chans];
    for (int chan = 0; chan < data.out_chans; chan++) {
      input_share_mac[chan] = new uint64_t[data.output_h * data.output_w];
      random_mod_p(prg, input_share_mac[chan],
                                 data.output_h * data.output_w, prime_mod);
    }
    vector<Plaintext> linear_mac = HE_preprocess_noise_plain(input_share_mac, data, *encoder_);

    //Preprocess Filter
    uint64_t*** filter_mat;
    filter_mat = (uint64_t***)malloc(sizeof(uint64_t**)*data.out_chans);
    for(int i=0; i<data.out_chans; i++) {
      filter_mat[i] = (uint64_t**)malloc(sizeof(uint64_t*)*data.inp_chans);
      for(int j=0; j< data.inp_chans; j++) {
        filter_mat[i][j] = (uint64_t*)malloc(sizeof(uint64_t)*data.filter_size);
        for(int k=0; k<data.filter_h; k++) {
          for(int l=0; l<data.filter_w; l++) {
            filter_mat[i][j][k*data.filter_w + l] = filterArr[k][l][j][i];
          }
        }
      }
    }

    auto masks_vec = HE_preprocess_filters(filter_mat, data, *encoder_);


    //Receive ciphertext
    vector<Ciphertext> ct_flat_rotations_input(data.inp_ct*data.filter_size);
    recv_encrypted_vector(io, context_, ct_flat_rotations_input);
    vector<vector<Ciphertext>> ct_vec(data.inp_ct, vector<Ciphertext>(data.filter_size));
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(int i=0; i<data.inp_ct; i++) {
      for(int j=0; j<data.filter_size; j++) {
        ct_vec[i][j] = ct_flat_rotations_input[i*data.filter_size+j];
      }
    }
    //Compute Convolution component
    auto conv_result = HE_conv(masks_vec, ct_vec, data, *evaluator_, *relin_keys_, *zero_);
    vector<Ciphertext> linear_ct = HE_output_rotations_dash(conv_result, data, *evaluator_, *gal_keys_,
            *zero_);
    //Generate Shares
    vector<Ciphertext> linear_ct_mac(data.out_ct);
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        // Linear MAC share
        evaluator_->multiply_plain(linear_ct[ct_idx], *enc_mac, linear_ct_mac[ct_idx]);
        // Linear share
        evaluator_->sub_plain_inplace(linear_ct[ct_idx], linear[ct_idx]);
        evaluator_->sub_plain_inplace(linear_ct_mac[ct_idx], linear_mac[ct_idx]);
    }
    //Send Ciphertexts
    send_encrypted_vector(io, linear_ct);
    send_encrypted_vector(io, linear_ct_mac);
  }
  cout<<"CP 3"<<endl;
}

void ConvField::convolution_gen(int32_t H, int32_t W, int32_t CI,
                            int32_t FH, int32_t FW, int32_t CO,
                            int32_t strideH, int32_t strideW, bool pad_valid,
                            vector<vector<vector<vector<uint64_t>>>> &inputArr,
                            vector<vector<vector<vector<uint64_t>>>> &inputMacArr,
                            vector<vector<vector<vector<uint64_t>>>> &filterArr,
                            Modulus mod,
                            bool verify_output, bool verbose) {
  data.image_h = H;
  data.image_w = W;
  data.inp_chans = CI;
  data.out_chans = CO;
  data.filter_h = FH;
  data.filter_w = FW;
  data.pad_valid = pad_valid;
  data.stride_h = strideH;
  data.stride_w = strideW;
  this->slot_count =
      min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, 2 * next_pow2(H * W)));
  configure_1();

  shared_ptr<SEALContext> context_;
  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  RelinKeys *relin_keys_;
  Ciphertext *zero_;
  if (slot_count == POLY_MOD_DEGREE) {
    context_ = this->context[0];
    encryptor_ = this->encryptor[0];
    decryptor_ = this->decryptor[0];
    evaluator_ = this->evaluator[0];
    encoder_ = this->encoder[0];
    gal_keys_ = this->gal_keys[0];
    relin_keys_ = this->relin_keys[0];
    zero_ = this->zero[0];
  } else if (slot_count == POLY_MOD_DEGREE_LARGE) {
    context_ = this->context[1];
    encryptor_ = this->encryptor[1];
    decryptor_ = this->decryptor[1];
    evaluator_ = this->evaluator[1];
    encoder_ = this->encoder[1];
    gal_keys_ = this->gal_keys[1];
    relin_keys_ = this->relin_keys[1];
    zero_ = this->zero[1];
  } else {
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_,
                      evaluator_, encoder_, gal_keys_, relin_keys_, zero_, verbose);
  }

  Image image;
  Image image_mac;
  Filters filters;

  image.resize(CI);
  for (int chan = 0; chan < CI; chan++) {
    Channel tmp_chan(H, W);
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        tmp_chan(h, w) = inputArr[0][h][w][chan];
      }
    }
    image[chan] = tmp_chan;
  }

  image_mac.resize(CI);
  for (int chan = 0; chan < CI; chan++) {
    Channel tmp_chan(H, W);
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        tmp_chan(h, w) = inputMacArr[0][h][w][chan];
      }
    }
    image_mac[chan] = tmp_chan;
  }

  auto pt = preprocess_image_OP(image, data);
  auto pt_mac = preprocess_image_OP(image_mac, data);

  if(party == BOB) {
    //Prepare and send ciphertext
    //auto ct_vec = HE_encrypt(pt, data, *encryptor_, *encoder_);
    auto rotated_pt = filter_rotations_dash(pt, data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, data, *encryptor_, *encoder_);
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());

    //auto ct_mac_vec = HE_encrypt(pt_mac, data, *encryptor_, *encoder_);

    auto rotated_pt_mac = filter_rotations_dash(pt_mac, data);
    auto ct_rotations_mac = HE_encrypt_rotations(rotated_pt_mac, data, *encryptor_, *encoder_);
    vector<Ciphertext> ct_flat_rotations_mac;
    for (const auto &ct: ct_rotations_mac)
        ct_flat_rotations_mac.insert(ct_flat_rotations_mac.end(), ct.begin(), ct.end());

    send_encrypted_vector(io, ct_flat_rotations);
    send_encrypted_vector(io, ct_flat_rotations_mac);

    /*auto pt = preprocess_image_OP(image, data);
    auto rotated_pt = filter_rotations_dash(pt, data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, data, *encryptor_, *encoder_);
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());
    cout<<"Ciphertext size is: "<<ct_flat_rotations.size()<<endl;
    cout<<"CT Ciphertext size is: "<<ct_rotations[0].size()<<endl;
    send_encrypted_vector(io, ct_flat_rotations);*/

    //Receive ciphertexts
    vector<Ciphertext> enc_inp_result(data.out_ct);
    vector<Ciphertext> enc_mac_result(data.out_ct);
    vector<Ciphertext> mac_ver_op(data.inp_ct);
    recv_encrypted_vector(io, context_, enc_inp_result);
    recv_encrypted_vector(io, context_, enc_mac_result);
    recv_encrypted_vector(io, context_, mac_ver_op);

    auto input_share_1 = HE_decrypt(enc_inp_result, data, *decryptor_, *encoder_);
    auto input_mac_share_1 = HE_decrypt(enc_mac_result, data, *decryptor_, *encoder_);

    //auto mac_ver = conv_preprocess_invert(mac_ver_op, data, *decryptor_, *encoder_);

  } else {

    auto rotated_pt_server = filter_rotations_dash(pt, data);
    auto rotated_pt_mac_server = filter_rotations_dash(pt_mac, data);



    vector<vector<Plaintext>> inp_plain(data.inp_ct, vector<Plaintext>(data.filter_size));
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
      for(int f_idx = 0 ; f_idx < data.filter_size; f_idx++) {
        encoder_->encode(rotated_pt_server[ct_idx][f_idx], inp_plain[ct_idx][f_idx]);
      }
    }

    vector<vector<Plaintext>> inp_mac_plain(data.inp_ct, vector<Plaintext>(data.filter_size));
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
      for(int f_idx = 0 ; f_idx < data.filter_size; f_idx++) {
        encoder_->encode(rotated_pt_mac_server[ct_idx][f_idx], inp_mac_plain[ct_idx][f_idx]);
      }
    }

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

    vector<vector<vector<vector<uint64_t>>>> input_mac_share(1);
    for (int i = 0; i < 1; i++) {
      input_mac_share[i].resize(data.image_h);
      for (int j = 0; j < data.image_h; j++) {
        input_mac_share[i][j].resize(data.image_w);
        for (int k = 0; k < W; k++) {
          input_mac_share[i][j][k].resize(data.inp_chans);
          random_mod_p(prg, input_mac_share[i][j][k].data(), data.inp_chans, prime_mod);
        }
      }
    }

    Image image_inp_mac;
    image_inp_mac.resize(data.inp_chans);
    for (int chan = 0; chan < data.inp_chans; chan++) {
      Channel tmp_chan(data.image_h, data.image_w);
      for (int h = 0; h < data.image_h; h++) {
        for (int w = 0; w < data.image_w; w++) {
          tmp_chan(h, w) = input_mac_share[0][h][w][chan];
        }
      }
      image_inp_mac[chan] = tmp_chan;
    }
    auto inp_mac_pt = preprocess_image_OP(image_inp_mac, data);

    //Output shares
    uint64_t **input_share = new uint64_t *[data.out_chans];
    for (int chan = 0; chan < data.out_chans; chan++) {
      input_share[chan] = new uint64_t[data.output_h * data.output_w];
      random_mod_p(prg, input_share[chan],
                                 data.output_h * data.output_w, prime_mod);
    }
    vector<Plaintext> linear = HE_preprocess_noise_plain(input_share, data, *encoder_);

    uint64_t **input_share_mac = new uint64_t *[data.out_chans];
    for (int chan = 0; chan < data.out_chans; chan++) {
      input_share_mac[chan] = new uint64_t[data.output_h * data.output_w];
      random_mod_p(prg, input_share_mac[chan],
                                 data.output_h * data.output_w, prime_mod);
    }
    vector<Plaintext> linear_mac = HE_preprocess_noise_plain(input_share_mac, data, *encoder_);

    //Preprocess Filter
    uint64_t*** filter_mat;
    filter_mat = (uint64_t***)malloc(sizeof(uint64_t**)*data.out_chans);
    for(int i=0; i<data.out_chans; i++) {
      filter_mat[i] = (uint64_t**)malloc(sizeof(uint64_t*)*data.inp_chans);
      for(int j=0; j< data.inp_chans; j++) {
        filter_mat[i][j] = (uint64_t*)malloc(sizeof(uint64_t)*data.filter_size);
        for(int k=0; k<data.filter_h; k++) {
          for(int l=0; l<data.filter_w; l++) {
            filter_mat[i][j][k*data.filter_w + l] = filterArr[k][l][j][i];
          }
        }
      }
    }

    auto masks_vec = HE_preprocess_filters(filter_mat, data, *encoder_);

    //Receive ciphertext
    vector<Ciphertext> ct_flat_rotations_input(data.inp_ct*data.filter_size);
    recv_encrypted_vector(io, context_, ct_flat_rotations_input);
    vector<vector<Ciphertext>> ct_vec(data.inp_ct, vector<Ciphertext>(data.filter_size));
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(int i=0; i<data.inp_ct; i++) {
      for(int j=0; j<data.filter_size; j++) {
        ct_vec[i][j] = ct_flat_rotations_input[i*data.filter_size+j];
      }
    }

    vector<Ciphertext> ct_flat_rotations_input_mac(data.inp_ct*data.filter_size);
    recv_encrypted_vector(io, context_, ct_flat_rotations_input_mac);
    vector<vector<Ciphertext>> ct_mac_vec(data.inp_ct, vector<Ciphertext>(data.filter_size));
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(int i=0; i<data.inp_ct; i++) {
      for(int j=0; j<data.filter_size; j++) {
        ct_mac_vec[i][j] = ct_flat_rotations_input_mac[i*data.filter_size+j];
      }
    }

    /*vector<Ciphertext> ct_vec(data.inp_ct);
    vector<Ciphertext> ct_mac_vec(data.inp_ct);
    recv_encrypted_vector(io, context_, ct_vec);
    recv_encrypted_vector(io, context_, ct_mac_vec);*/

    vector<vector<Ciphertext>> ct_vec_dash(data.inp_ct, vector<Ciphertext>(data.filter_size));
    vector<vector<Ciphertext>> ct_mac_vec_dash(data.inp_ct, vector<Ciphertext>(data.filter_size));

    //vector<vector<Plaintext>> inp_plain_rotate = filter_rotations_dash(inp_plain, data, evaluator_, gal_keys_);
    //vector<vector<Plaintext>> inp_plain_mac_rotate = filter_rotations_dash(inp_mac_plain, data, evaluator_, gal_keys_);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        for(int f_idx = 0; f_idx < data.filter_size; f_idx++) {
          evaluator_->add_plain(ct_vec[ct_idx][f_idx], inp_plain[ct_idx][f_idx], ct_vec_dash[ct_idx][f_idx]);
          evaluator_->add_plain(ct_mac_vec[ct_idx][f_idx], inp_mac_plain[ct_idx][f_idx], ct_mac_vec_dash[ct_idx][f_idx]);
        }
    }
    /*
    vector<Ciphertext> input_ct(data.inp_ct);
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        evaluator_->add_plain(ct_vec[ct_idx], inp_plain[ct_idx], input_ct[ct_idx]);
    }

    vector<Ciphertext> input_mac_ct(data.inp_ct);
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        evaluator_->add_plain(ct_mac_vec[ct_idx], inp_mac_plain[ct_idx], input_mac_ct[ct_idx]);
    }

    vector<vector<Ciphertext>> input_rotations(data.inp_ct);
    for (int i = 0; i < data.inp_ct; i++) {
      input_rotations[i].resize(data.filter_size);
    }
    input_rotations = filter_rotations_dash(input_ct, data, evaluator_, gal_keys_);

    vector<vector<Ciphertext>> input_mac_rotations(data.inp_ct);
    for (int i = 0; i < data.inp_ct; i++) {
      input_mac_rotations[i].resize(data.filter_size);
    }
    input_mac_rotations = filter_rotations_dash(input_mac_ct, data, evaluator_, gal_keys_);
    */
    /*vector<vector<Ciphertext>> ct_vec(data.inp_ct, vector<Ciphertext>(data.filter_size));
    for(int i=0; i<data.inp_ct; i++) {
      for(int j=0; j<data.filter_size; j++) {
        ct_vec[i][j] = ct_flat_rotations_input[i*data.filter_size+j];
      }
    }

    vector<vector<Ciphertext>> ct_mac_vec(data.inp_ct, vector<Ciphertext>(data.filter_size));
    for(int i=0; i<data.inp_ct; i++) {
      for(int j=0; j<data.filter_size; j++) {
        ct_mac_vec[i][j] = ct_flat_rotations_input_mac[i*data.filter_size+j];
      }
    }

    int zero_idx = 0;
    int offset = (data.image_w + data.pad_l + data.pad_r - data.filter_w + 1) * data.pad_t + data.pad_l;
    if (offset != 0) {
        zero_idx = (data.filter_size - 1) / 2;
    }

    vector<Ciphertext> input_ct(data.inp_ct);
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        evaluator_->add_plain(ct_vec[ct_idx][zero_idx], inp_plain[ct_idx], input_ct[ct_idx]);
    }

    vector<Ciphertext> input_mac_ct(data.inp_ct);
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        evaluator_->add_plain(ct_mac_vec[ct_idx][zero_idx], inp_mac_plain[ct_idx], input_mac_ct[ct_idx]);
    }*/

    //Reconstruct inputs

    auto conv_result = HE_conv(masks_vec, ct_vec_dash, data, *evaluator_, *relin_keys_, *zero_);
    auto conv_result_mac = HE_conv(masks_vec, ct_mac_vec_dash, data, *evaluator_, *relin_keys_, *zero_);
    vector<Ciphertext> linear_ct = HE_output_rotations_dash(conv_result, data, *evaluator_, *gal_keys_,
            *zero_);
    vector<Ciphertext> linear_mac_ct = HE_output_rotations_dash(conv_result_mac, data, *evaluator_, *gal_keys_,
                    *zero_);

    vector<Ciphertext> mac_ver_op_cube(data.inp_ct);
    vector<Ciphertext> mac_ver_op_sqrt(data.inp_ct);
    vector<Ciphertext> mac_ver_op(data.inp_ct);
    int zero_idx = 0;
    int offset = (data.image_w + data.pad_l + data.pad_r - data.filter_w + 1) * data.pad_t + data.pad_l;
    if (offset != 0) {
        zero_idx = (data.filter_size - 1) / 2;
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
      evaluator_->multiply_plain(ct_vec_dash[ct_idx][zero_idx], *enc_mac_cube, mac_ver_op_cube[ct_idx]);
      evaluator_->multiply_plain(ct_mac_vec_dash[ct_idx][zero_idx], *enc_mac_sqrt, mac_ver_op_sqrt[ct_idx]);
      evaluator_->sub(mac_ver_op_cube[ct_idx], mac_ver_op_sqrt[ct_idx], mac_ver_op[ct_idx]);
    }
    //Generate Shares
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        // Linear share
        evaluator_->sub_plain_inplace(linear_ct[ct_idx], linear[ct_idx]);
        evaluator_->sub_plain_inplace(linear_mac_ct[ct_idx], linear_mac[ct_idx]);
    }

    //Send Ciphertexts
    send_encrypted_vector(io, linear_ct);
    send_encrypted_vector(io, linear_mac_ct);
    send_encrypted_vector(io, mac_ver_op);
  }

}

void ConvField::convolution(int32_t N, int32_t H, int32_t W, int32_t CI,
                            int32_t FH, int32_t FW, int32_t CO,
                            int32_t zPadHLeft, int32_t zPadHRight,
                            int32_t zPadWLeft, int32_t zPadWRight,
                            int32_t strideH, int32_t strideW,
                            vector<vector<vector<vector<uint64_t>>>> &inputArr,
                            vector<vector<vector<vector<uint64_t>>>> &filterArr,
                            vector<vector<vector<vector<uint64_t>>>> &outArr,
                            bool verify_output, bool verbose) {
  int paddedH = H + zPadHLeft + zPadHRight;
  int paddedW = W + zPadWLeft + zPadWRight;
  int newH = 1 + (paddedH - FH) / strideH;
  int newW = 1 + (paddedW - FW) / strideW;
  int limitH = FH + ((paddedH - FH) / strideH) * strideH;
  int limitW = FW + ((paddedW - FW) / strideW) * strideW;

  for (int i = 0; i < newH; i++) {
    for (int j = 0; j < newW; j++) {
      for (int k = 0; k < CO; k++) {
        outArr[0][i][j][k] = 0;
      }
    }
  }

  Image image;
  Filters filters;

  image.resize(CI);
  for (int chan = 0; chan < CI; chan++) {
    Channel tmp_chan(H, W);
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        tmp_chan(h, w) =
            neg_mod((int64_t)inputArr[0][h][w][chan], (int64_t)prime_mod);
      }
    }
    image[chan] = tmp_chan;
  }
  if (party == BOB) {
    for (int s_row = 0; s_row < strideH; s_row++) {
      for (int s_col = 0; s_col < strideW; s_col++) {
        int lH = ((limitH - s_row + strideH - 1) / strideH);
        int lW = ((limitW - s_col + strideW - 1) / strideW);
        int lFH = ((FH - s_row + strideH - 1) / strideH);
        int lFW = ((FW - s_col + strideW - 1) / strideW);
        Image lImage(CI);
        for (int chan = 0; chan < CI; chan++) {
          Channel tmp_chan(lH, lW);
          // lImage[chan] = new uint64_t[lH*lW];
          for (int row = 0; row < lH; row++) {
            for (int col = 0; col < lW; col++) {
              int idxH = row * strideH + s_row - zPadHLeft;
              int idxW = col * strideW + s_col - zPadWLeft;
              if ((idxH < 0 || idxH >= H) || (idxW < 0 || idxW >= W)) {
                tmp_chan(row, col) = 0;
              } else {
                tmp_chan(row, col) =
                    neg_mod(inputArr[0][idxH][idxW][chan], (int64_t)prime_mod);
              }
            }
          }
          lImage[chan] = tmp_chan;
        }
        if (lFH > 0 && lFW > 0) {
          non_strided_conv(lH, lW, CI, lFH, lFW, CO, &lImage, nullptr,
                           outArr[0], verbose);
        }
      }
    }
    for (int idx = 0; idx < newH * newW; idx++) {
      for (int chan = 0; chan < CO; chan++) {
        outArr[0][idx / newW][idx % newW][chan] = neg_mod(
            (int64_t)outArr[0][idx / newW][idx % newW][chan], prime_mod);
      }
    }
    if (verify_output)
      verify(H, W, CI, CO, image, nullptr, outArr);
  } else // party == ALICE
  {
    filters.resize(CO);
    for (int out_c = 0; out_c < CO; out_c++) {
      Image tmp_img(CI);
      for (int inp_c = 0; inp_c < CI; inp_c++) {
        Channel tmp_chan(FH, FW);
        for (int idx = 0; idx < FH * FW; idx++) {
          tmp_chan(idx / FW, idx % FW) =
              (int64_t)filterArr[idx / FW][idx % FW][inp_c][out_c];
        }
        tmp_img[inp_c] = tmp_chan;
      }
      filters[out_c] = tmp_img;
    }

    for (int s_row = 0; s_row < strideH; s_row++) {
      for (int s_col = 0; s_col < strideW; s_col++) {
        int lH = ((limitH - s_row + strideH - 1) / strideH);
        int lW = ((limitW - s_col + strideW - 1) / strideW);
        int lFH = ((FH - s_row + strideH - 1) / strideH);
        int lFW = ((FW - s_col + strideW - 1) / strideW);
        Filters lFilters(CO);
        for (int out_c = 0; out_c < CO; out_c++) {
          Image tmp_img(CI);
          for (int inp_c = 0; inp_c < CI; inp_c++) {
            Channel tmp_chan(lFH, lFW);
            for (int row = 0; row < lFH; row++) {
              for (int col = 0; col < lFW; col++) {
                int idxFH = row * strideH + s_row;
                int idxFW = col * strideW + s_col;
                tmp_chan(row, col) = neg_mod(
                    filterArr[idxFH][idxFW][inp_c][out_c], (int64_t)prime_mod);
              }
            }
            tmp_img[inp_c] = tmp_chan;
          }
          lFilters[out_c] = tmp_img;
        }
        if (lFH > 0 && lFW > 0) {
          non_strided_conv(lH, lW, CI, lFH, lFW, CO, nullptr, &lFilters,
                           outArr[0], verbose);
        }
      }
    }
    data.image_h = H;
    data.image_w = W;
    data.inp_chans = CI;
    data.out_chans = CO;
    data.filter_h = FH;
    data.filter_w = FW;
    data.pad_t = zPadHLeft;
    data.pad_b = zPadHRight;
    data.pad_l = zPadWLeft;
    data.pad_r = zPadWRight;
    data.stride_h = strideH;
    data.stride_w = strideW;

    // The filter values should be small enough to not overflow uint64_t
    Image local_result = ideal_functionality(image, filters);

    for (int idx = 0; idx < newH * newW; idx++) {
      for (int chan = 0; chan < CO; chan++) {
        outArr[0][idx / newW][idx % newW][chan] =
            neg_mod((int64_t)local_result[chan](idx / newW, idx % newW) +
                        (int64_t)outArr[0][idx / newW][idx % newW][chan],
                    prime_mod);
      }
    }
    if (verify_output)
      verify(H, W, CI, CO, image, &filters, outArr);
  }
}

void ConvField::verify(int H, int W, int CI, int CO, Image &image,
                       Filters *filters,
                       vector<vector<vector<vector<uint64_t>>>> &outArr) {
  int newH = outArr[0].size();
  int newW = outArr[0][0].size();
  if (party == BOB) {
    for (int i = 0; i < CI; i++) {
      io->send_data(image[i].data(), H * W * sizeof(uint64_t));
    }
    for (int i = 0; i < newH; i++) {
      for (int j = 0; j < newW; j++) {
        io->send_data(outArr[0][i][j].data(),
                      sizeof(uint64_t) * data.out_chans);
      }
    }
  } else // party == ALICE
  {
    Image image_0(CI); // = new Channel[CI];
    for (int i = 0; i < CI; i++) {
      // image_0[i] = new uint64_t[H*W];
      image_0[i].resize(H, W);
      io->recv_data(image_0[i].data(), H * W * sizeof(uint64_t));
    }
    for (int i = 0; i < CI; i++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          image[i](h, w) = (image[i](h, w) + image_0[i](h, w)) % prime_mod;
        }
      }
    }
    Image result = ideal_functionality(image, *filters);

    vector<vector<vector<vector<uint64_t>>>> outArr_0;
    outArr_0.resize(1);
    outArr_0[0].resize(newH);
    for (int i = 0; i < newH; i++) {
      outArr_0[0][i].resize(newW);
      for (int j = 0; j < newW; j++) {
        outArr_0[0][i][j].resize(CO);
        io->recv_data(outArr_0[0][i][j].data(), sizeof(uint64_t) * CO);
      }
    }
    for (int i = 0; i < newH; i++) {
      for (int j = 0; j < newW; j++) {
        for (int k = 0; k < CO; k++) {
          outArr_0[0][i][j][k] =
              (outArr_0[0][i][j][k] + outArr[0][i][j][k]) % prime_mod;
        }
      }
    }
    bool pass = true;
    for (int i = 0; i < CO; i++) {
      for (int j = 0; j < newH; j++) {
        for (int k = 0; k < newW; k++) {
          if ((int64_t)outArr_0[0][j][k][i] !=
              neg_mod(result[i](j, k), (int64_t)prime_mod)) {
            pass = false;
          }
        }
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
    }
  }
}
