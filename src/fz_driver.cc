#include "fz_driver.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace fzgpu {

void check_cuda_error(cudaError_t status, const char* file, int line)
{
  if (cudaSuccess != status) {
    printf("\n");
    printf(
        "CUDA API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n",  //
        file, line, cudaGetErrorString(status), status);
    exit(EXIT_FAILURE);
  }
}

internal_membuf::internal_membuf(config_map config, bool _verifiy_on) :
    verify_on(_verifiy_on)
{
  auto grid_x = config["grid_x"];
  auto pad_len = config["pad_len"];
  auto chunk_size = config["chunk_size"];

  // CHECK_CUDA2(cudaMalloc(&d_input, sizeof(float) * pad_len));
  CHECK_CUDA2(cudaMalloc(&d_quantcode, sizeof(uint16_t) * pad_len));
  CHECK_CUDA2(cudaMalloc(&d_signum, sizeof(bool) * pad_len));
  CHECK_CUDA2(cudaMalloc(&d_comp_out, sizeof(uint16_t) * pad_len));
  CHECK_CUDA2(cudaMalloc(&d_bitflag_array, sizeof(uint32_t) * chunk_size));
  CHECK_CUDA2(cudaMalloc(&d_decomp_quantcode, sizeof(uint16_t) * pad_len));
  //   CHECK_CUDA2(cudaMalloc(&d_decomp_output, sizeof(float) * pad_len));

  CHECK_CUDA2(cudaMalloc(&d_offset_counter, sizeof(uint32_t)));
  CHECK_CUDA2(cudaMalloc(&d_start_pos, sizeof(uint32_t) * grid_x));
  CHECK_CUDA2(cudaMalloc(&d_comp_size, sizeof(uint32_t) * grid_x));

  CHECK_CUDA2(cudaMemset(d_quantcode, 0, sizeof(uint16_t) * pad_len));
  CHECK_CUDA2(cudaMemset(d_bitflag_array, 0, sizeof(uint32_t) * chunk_size));
  CHECK_CUDA2(cudaMemset(d_decomp_quantcode, 0, sizeof(uint16_t) * pad_len));

  CHECK_CUDA2(cudaMemset(d_offset_counter, 0, sizeof(uint32_t)));
  CHECK_CUDA2(cudaMemset(d_start_pos, 0, sizeof(uint32_t) * grid_x));
  CHECK_CUDA2(cudaMemset(d_comp_size, 0, sizeof(uint32_t) * grid_x));
  //   CHECK_CUDA2(cudaMemset(d_decomp_output, 0, sizeof(float) * pad_len));

  if (verify_on) {
    h_quantcode = new uint16_t[config["len"]];
    h_decomp_quantcode = new uint16_t[config["len"]];
    h_decomp_output = new float[config["len"]];
  }
}

internal_membuf::~internal_membuf()
{
  // CHECK_CUDA2(cudaFree(d_input));
  CHECK_CUDA2(cudaFree(d_quantcode));
  CHECK_CUDA2(cudaFree(d_signum));
  CHECK_CUDA2(cudaFree(d_comp_out));
  CHECK_CUDA2(cudaFree(d_bitflag_array));

  CHECK_CUDA2(cudaFree(d_offset_counter));
  CHECK_CUDA2(cudaFree(d_start_pos));
  CHECK_CUDA2(cudaFree(d_comp_size));
  CHECK_CUDA2(cudaFree(d_decomp_quantcode));
  //   CHECK_CUDA2(cudaFree(d_decomp_output));

  if (verify_on) {
    delete[] h_quantcode;
    delete[] h_decomp_quantcode;
    delete[] h_decomp_output;
  }
}

}  // namespace fzgpu

namespace fzgpu {

Compressor::Compressor(int const x, int const y, int const z) :
    config(fzgpu::utils::configure_fzgpu(x, y, z)),
    len3(dim3(x, y, z)),
    x(x),
    y(y),
    z(z)
{
  buf = new fzgpu::internal_membuf(config);
}

Compressor::~Compressor() { delete buf; }

void Compressor::profile_data_range(
    float* h_input, size_t const len, double& range)
{
  range = *std::max_element(h_input, h_input + len) -
          *std::min_element(h_input, h_input + len);
}

void Compressor::postenc_make_offsetsum_on_host()
{
  CHECK_CUDA2(cudaMemcpy(
      &membuf()->h_offset_sum, membuf()->d_offset_counter, sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
}

size_t Compressor::postenc_calc_compressed_size() const
{
  return sizeof(uint32_t) * config.at("chunk_size") +  // d_bitflag_array
         membuf()->h_offset_sum * sizeof(uint32_t) +   // partially d_comp_out
         sizeof(uint32_t) *
             int(this->config.at("quantcode_bytes") / 4096);  // start_pos
}

void Compressor::compress(
    float* in, double eb, uint16_t** pd_archive, size_t* archive_size,
    void* stream)
{
  float _time;
  comp_start = std::chrono::system_clock::now();

  fzgpu::cuhip::GPU_lorenzo_predict_fz_variant(
      buf->d_input, buf->d_quantcode, buf->d_signum, len3, eb, _time,
      (cudaStream_t)stream);

  fzgpu::cuhip::GPU_FZ_encode(
      buf->d_quantcode, buf->d_comp_out, buf->d_offset_counter,
      buf->d_bitflag_array, buf->d_start_pos, buf->d_comp_size, x, y, z,
      (cudaStream_t)stream);

  cudaStreamSynchronize((cudaStream_t)stream);
  comp_end = std::chrono::system_clock::now();

  *pd_archive = buf->d_comp_out;

  postenc_make_offsetsum_on_host();
  *archive_size = postenc_calc_compressed_size();
}

void Compressor::decompress(
    uint16_t* d_archive, double const eb, float* d_out, void* stream)
{
  float _time;
  decomp_start = std::chrono::system_clock::now();

  fzgpu::cuhip::GPU_FZ_decode(
      d_archive, buf->d_decomp_quantcode, buf->d_bitflag_array,
      buf->d_start_pos, x, y, z, (cudaStream_t)stream);

  fzgpu::cuhip::GPU_reverse_lorenzo_predict_fz_variant(
      buf->d_signum, buf->d_decomp_quantcode, d_out, len3, eb, _time,
      (cudaStream_t)stream);

  cudaStreamSynchronize((cudaStream_t)stream);
  decomp_end = std::chrono::system_clock::now();
}

void fzgpu_cli_compressor_roundtrip(
    std::string fname, int const x, int const y, int const z, double eb)
{
}

void Compressor::postdecomp_verification(
    float const* d_out, double const eb, cudaStream_t stream)
{
  CHECK_CUDA2(cudaMemcpy(
      membuf()->h_quantcode, membuf()->d_quantcode,
      sizeof(uint16_t) * config.at("len"), cudaMemcpyDeviceToHost));

  // bitshuffle verification
  CHECK_CUDA2(cudaMemcpy(
      membuf()->h_decomp_quantcode, membuf()->d_decomp_quantcode,
      sizeof(uint16_t) * config.at("len"), cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  auto bitshuffle_verify = fzgpu::utils::bitshuffle_verify(
      membuf()->h_quantcode, membuf()->h_decomp_quantcode, config.at("len"));

  // prequant verification
  CHECK_CUDA2(cudaMemcpy(
      membuf()->h_decomp_output, d_out, sizeof(float) * config.at("len"),
      cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  bool prequant_verify = true;
  if (bitshuffle_verify)
    prequant_verify = fzgpu::utils::prequantization_verify(
        membuf()->h_input, membuf()->h_decomp_output, config.at("len"), eb);

  fzgpu::utils::verify_data<float>(
      membuf()->h_decomp_output, membuf()->h_input, config.at("len"));

  // print verification result
  if (bitshuffle_verify) {
    printf("%-*s%*s\n", 30, "verify::bitshuffle", 10, "ok");
    if (prequant_verify)
      printf("%-*s%*s\n", 30, "verify::prequant", 10, "ok");
    else
      printf("%-*s%*s\n", 30, "verify::prequant", 10, "FAIL");
  }
  else {
    printf("%-*s%*s\n", 30, "verify::bitshuffle", 10, "FAIL");
  }
}

}  // namespace fzgpu
