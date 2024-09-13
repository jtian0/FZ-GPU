#include <cstdint>

#include "fz_module.hh"
#include "kernel/lorenzo_var.cuh"

namespace fzgpu {

#include "fz.cu"

}

namespace fzgpu::utils {

config_map configure_fzgpu(int const x, int const y, int const z)
{
  constexpr auto UINT32_BIT_LEN = 32;
  auto const dataTypeLen = x * y * z;
  auto blockSize = 16;
  auto quantizationCodeByteLen =
      dataTypeLen * 2;  // quantization code length in unit of bytes
  quantizationCodeByteLen =
      quantizationCodeByteLen % 4096 == 0
          ? quantizationCodeByteLen
          : quantizationCodeByteLen - quantizationCodeByteLen % 4096 + 4096;
  auto paddingDataTypeLen = quantizationCodeByteLen / 2;
  int dataChunkSize =
      quantizationCodeByteLen % (blockSize * UINT32_BIT_LEN) == 0
          ? quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN)
          : int(quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN)) + 1;
//   cout << "len\t" << dataTypeLen << endl;
//   cout << "bytes\t" << dataTypeLen * sizeof(float) << endl;
//   cout << "pad_len\t" << paddingDataTypeLen << endl;
//   cout << "chunk_size\t" << dataChunkSize << endl;
//   cout << "quantcode_bytes\t" << quantizationCodeByteLen << endl;
//   cout << "grid_x\t" << floor(quantizationCodeByteLen / 4096) << endl;

  //   dim3 grid(floor(paddingDataTypeLen / 2048));
  return config_map{
      {"len", dataTypeLen},
      {"bytes", dataTypeLen * sizeof(float)},
      {"pad_len", paddingDataTypeLen},
      {"chunk_size", dataChunkSize},
      {"quantcode_bytes", quantizationCodeByteLen},
      //   {"grid_x", floor(paddingDataTypeLen / 2048)}
      {"grid_x", floor(quantizationCodeByteLen / 4096)}};
}
}  // namespace fzgpu::utils

namespace fzgpu::cuhip {

using T = float;
using E = uint16_t;
using FP = float;

fzgpuerror GPU_lorenzo_predict_fz_variant(
    T* d_input, E* d_quantcode, bool* d_signum, dim3 const input_dim3,
    double const eb, float& time_elapsed, cudaStream_t stream)
{
  psz::cuhip::launch_construct_LorenzoI_var<T, E, FP>(
      d_input, d_quantcode, d_signum, input_dim3, eb, time_elapsed, stream);

  return FZGPU_SUCCESS;
}

fzgpuerror GPU_reverse_lorenzo_predict_fz_variant(
    bool* d_signum, E* d_quantcode, T* d_decomp_output, dim3 const input_dim3,
    double const eb, float& time_elapsed, cudaStream_t stream)
{
  psz::cuhip::launch_reconstruct_LorenzoI_var<float, uint16_t, float>(
      d_signum, d_quantcode, d_decomp_output, input_dim3, eb, time_elapsed,
      stream);

  return FZGPU_SUCCESS;
}

fzgpuerror GPU_FZ_encode(
    uint16_t* d_quantcode, uint16_t* d_comp_out, uint32_t* d_offset_counter,
    uint32_t* d_bitflag_array, uint32_t* d_start_pos, uint32_t* d_comp_size,
    int const x, int const y, int const z, cudaStream_t stream)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  dim3 grid = dim3(config["grid_x"]);
  dim3 block(32, 32);

  fzgpu::KERNEL_CUHIP_fz_fused_encode<<<grid, block, 0, stream>>>(
      (uint32_t*)d_quantcode, (uint32_t*)d_comp_out, d_offset_counter,
      d_bitflag_array, d_start_pos, d_comp_size);

//   cudaStreamSynchronize(stream);

  return FZGPU_SUCCESS;
}

fzgpuerror GPU_FZ_decode(
    uint16_t* d_comp_out, uint16_t* d_decomp_quantcode,
    uint32_t* d_bitflag_array, uint32_t* d_start_pos, int const x, int const y,
    int const z, cudaStream_t stream)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  dim3 grid = dim3(config["grid_x"]);
  dim3 block(32, 32);

  fzgpu::KERNEL_CUHIP_fz_fused_decode<<<grid, block, 0, stream>>>(
      (uint32_t*)d_comp_out, (uint32_t*)d_decomp_quantcode, d_bitflag_array,
      d_start_pos);

  cudaStreamSynchronize(stream);

  return FZGPU_SUCCESS;
}

}  // namespace fzgpu::cuhip