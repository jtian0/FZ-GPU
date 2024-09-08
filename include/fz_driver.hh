#ifndef AAA05CE8_BA0A_4271_B0A1_8ACDD51AFF1F
#define AAA05CE8_BA0A_4271_B0A1_8ACDD51AFF1F

#include <chrono>
#include <cstdint>

#include "fz_module.hh"

namespace fzgpu {

using time_t = std::chrono::time_point<std::chrono::system_clock>;

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

}  // namespace fzgpu

namespace fzgpu::utils {

template <typename T>
void verify_data(T* xdata, T* odata, size_t len);
}

namespace fzgpu {

void fzgpu_compressor_roundtrip(
    std::string fname, int const x, int const y, int const z, double eb);

}

#endif /* AAA05CE8_BA0A_4271_B0A1_8ACDD51AFF1F */
