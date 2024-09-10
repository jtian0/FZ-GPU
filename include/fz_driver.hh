#ifndef AAA05CE8_BA0A_4271_B0A1_8ACDD51AFF1F
#define AAA05CE8_BA0A_4271_B0A1_8ACDD51AFF1F

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "fz_module.hh"

namespace fzgpu {

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::duration<double>;

class internal_membuf {
  bool verify_on;

 public:
  // round trip (comp + decomp altogether)
  float* d_input;
  float* h_input;
  uint32_t h_offset_sum;

  uint16_t* d_comp_out;
  uint32_t* d_bitflag_array;
  uint16_t* d_quantcode;
  uint32_t* d_offset_counter;
  uint32_t* d_start_pos;
  uint32_t* d_comp_size;
  uint16_t* d_decomp_quantcode;
  float* d_decomp_output;
  bool* d_signum;

  // verification only
  uint16_t* h_quantcode;
  uint16_t* h_decomp_quantcode;
  float* h_decomp_output;

  internal_membuf(fzgpu::config_map config, bool verify_on = true);
  ~internal_membuf();
};

class Compressor {
 private:
  fzgpu::config_map const config;
  fzgpu::internal_membuf* buf;
  int const x, y, z;

  dim3 len3;  // TODO use vendor-agnostic type

  void postenc_make_offsetsum_on_host();
  size_t postenc_calc_compressed_size();

 public:
  fzgpu::time_t comp_start, comp_end, decomp_start, decomp_end;

  using T = float;
  using E = uint16_t;
  Compressor(int const x, int const y, int const z);
  ~Compressor();
  static void profile_data_range(
      float* h_input, size_t const len, double& range);
  void compress(
      float* d_in, double const eb, uint16_t** pd_archive, size_t* archive_size, void* stream);
  void decompress(uint16_t* d_archive, double const eb, void* stream);

  // getter (testing purposes)
  float*& input_hptr() const { return buf->h_input; };
  float*& input_dptr() const { return buf->d_input; };
  fzgpu::internal_membuf* const& membuf() const { return buf; };
};

}  // namespace fzgpu

namespace fzgpu {

void fzgpu_reference_compressor_roundtrip(
    std::string fname, int const x, int const y, int const z, double eb);

void fzgpu_compressor_roundtrip_v1(
    std::string fname, int const x, int const y, int const z, double eb);

}  // namespace fzgpu

#endif /* AAA05CE8_BA0A_4271_B0A1_8ACDD51AFF1F */
