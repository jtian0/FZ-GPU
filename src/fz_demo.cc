#include "fz_demo.hh"

#include <algorithm>
#include <iostream>
#include <string>

#include "fz_driver.hh"
#include "fz_module.hh"
#include "fz_utils.hh"
#include "utils/io.hh"

using namespace fzgpu;

// TODO move to example/test
void fzgpu::compressor_roundtrip_v0(
    std::string fname, int const x, int const y, int const z, double eb,
    bool use_rel)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  auto buf = fzgpu::internal_membuf(config);

  auto len3 = dim3(x, y, z);

  float time_elapsed;

  fzgpu::time_t comp_start, comp_end, decomp_start, decomp_end;

  buf.h_input = io::read_binary_to_new_array<float>(fname, config["pad_len"]);
  CHECK_CUDA2(cudaMalloc(&buf.d_input, sizeof(float) * config["len"]));
  CHECK_CUDA2(cudaMemset(buf.d_input, 0x0, sizeof(float) * config["len"]));

  float* d_decompressed;
  CHECK_CUDA2(cudaMalloc(&d_decompressed, sizeof(float) * config["pad_len"]));

  if (use_rel) {
    double range =
        *std::max_element(buf.h_input, buf.h_input + config["pad_len"]) -
        *std::min_element(buf.h_input, buf.h_input + config["pad_len"]);
    eb *= range;
  }

  CHECK_CUDA2(cudaMemcpy(
      buf.d_input, buf.h_input, sizeof(float) * config["len"],
      cudaMemcpyHostToDevice));

  //// data is ready
  ////////////////////////////////////////////////////////////////

  cudaStream_t stream;  // external to compressor
  cudaStreamCreate(&stream);

  comp_start = std::chrono::system_clock::now();

  fzgpu::cuhip::GPU_lorenzo_predict_fz_variant(
      buf.d_input, buf.d_quantcode, buf.d_signum, len3, eb, time_elapsed,
      stream);

  fzgpu::cuhip::GPU_FZ_encode(
      buf.d_quantcode, buf.d_comp_out, buf.d_offset_counter,
      buf.d_bitflag_array, buf.d_start_pos, buf.d_comp_size, x, y, z, stream);

  cudaStreamSynchronize(stream);
  comp_end = std::chrono::system_clock::now();

  decomp_start = std::chrono::system_clock::now();

  fzgpu::cuhip::GPU_FZ_decode(
      buf.d_comp_out, buf.d_decomp_quantcode, buf.d_bitflag_array,
      buf.d_start_pos, x, y, z, stream);

  fzgpu::cuhip::GPU_reverse_lorenzo_predict_fz_variant(
      buf.d_signum, buf.d_decomp_quantcode, d_decompressed, len3, eb,
      time_elapsed, stream);

  cudaStreamSynchronize(stream);
  decomp_end = std::chrono::system_clock::now();

  ////////////////////////////////////////////////////////////////////////////////

  CHECK_CUDA2(cudaMemcpy(
      buf.h_quantcode, buf.d_quantcode, sizeof(uint16_t) * config["len"],
      cudaMemcpyDeviceToHost));

  // bitshuffle verification
  CHECK_CUDA2(cudaMemcpy(
      buf.h_decomp_quantcode, buf.d_decomp_quantcode,
      sizeof(uint16_t) * config["len"], cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  auto bitshuffle_verify = fzgpu::utils::bitshuffle_verify(
      buf.h_quantcode, buf.h_decomp_quantcode, config["len"]);

  // prequant verification
  CHECK_CUDA2(cudaMemcpy(
      buf.h_decomp_output, d_decompressed, sizeof(float) * config["len"],
      cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  bool prequant_verify = true;
  if (bitshuffle_verify)
    prequant_verify = fzgpu::utils::prequantization_verify(
        buf.h_input, buf.h_decomp_output, config["len"], eb);

  fzgpu::utils::verify_data<float>(
      buf.h_decomp_output, buf.h_input, config["len"]);

  // print verification result
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

  ////////////////////////////////////////////////////////////////////////////////

  CHECK_CUDA2(cudaMemcpy(
      &buf.h_offset_sum, buf.d_offset_counter, sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  printf("%-*s%10ld\n", 30, "data::original_bytes", config["bytes"]);
  auto comp_size = sizeof(uint32_t) * config["chunk_size"] +
                   buf.h_offset_sum * sizeof(uint32_t) +
                   sizeof(uint32_t) * int(config["quantcode_bytes"] / 4096);
  printf("%-*s%10ld\n", 30, "data::comp_bytes", comp_size);
  printf(
      "%-*s%10.2f\n", 30, "comp_metric::CR",
      float(config["bytes"]) / float(comp_size));

  duration_t comp_time = comp_end - comp_start;
  duration_t decomp_time = decomp_end - decomp_start;

  fzgpu::utils::print_speed(
      comp_time.count(), decomp_time.count(), config["bytes"]);

  ////////////////////////////////////////////////////////////////////////////////

  cudaStreamDestroy(stream);

  cudaFree(buf.d_input);
  cudaFree(d_decompressed);
  delete[] buf.h_input;

  return;
}

void fzgpu::compressor_roundtrip_v1(
    std::string fname, int const x, int const y, int const z, double eb,
    bool use_rel)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  Compressor cor(x, y, z);
  double range;

  cor.input_hptr() =
      io::read_binary_to_new_array<float>(fname, config["pad_len"]);
  CHECK_CUDA2(cudaMalloc(&cor.input_dptr(), sizeof(float) * config["len"]));
  CHECK_CUDA2(cudaMemset(cor.input_dptr(), 0, sizeof(float) * config["len"]));

  if (use_rel) {
    Compressor::profile_data_range(cor.input_hptr(), config["pad_len"], range);
    eb *= range;
  }

  CHECK_CUDA2(cudaMemcpy(
      cor.input_dptr(), cor.input_hptr(), sizeof(float) * config["len"],
      cudaMemcpyHostToDevice));

  uint16_t* d_compressed_internal{nullptr};
  size_t compressed_size;

  cudaStream_t stream;  // external to compressor
  cudaStreamCreate(&stream);
  //// data is ready
  cor.compress(
      cor.input_dptr(), eb, &d_compressed_internal, &compressed_size, stream);

  //// mimick copying out compressed data after compression
  uint16_t* d_compressed_dump;
  CHECK_CUDA2(
      cudaMalloc(&d_compressed_dump, compressed_size * sizeof(uint16_t)));
  CHECK_CUDA2(cudaMemcpy(
      d_compressed_dump, d_compressed_internal,
      sizeof(uint16_t) * compressed_size, cudaMemcpyDeviceToDevice));

  ////  mimick allocating for decompressed data before decompression
  float* d_decompressed;
  CHECK_CUDA2(cudaMalloc(&d_decompressed, sizeof(float) * config["pad_len"]));

  //// decompress using external saved archive
  cor.decompress(d_compressed_dump, eb, d_decompressed, stream);

  cor.postdecomp_verification(d_decompressed, eb, stream);

  //// print log
  printf("%-*s%10ld\n", 30, "data::original_bytes", config["bytes"]);
  auto comp_size = cor.comp_size();
  printf("%-*s%10ld\n", 30, "data::comp_bytes", comp_size);
  printf(
      "%-*s%10.2f\n", 30, "comp_metric::CR",
      float(config["bytes"]) / float(comp_size));

  duration_t comp_time = cor.comp_end - cor.comp_start;
  duration_t decomp_time = cor.decomp_end - cor.decomp_start;

  fzgpu::utils::print_speed(
      comp_time.count(), decomp_time.count(), config["bytes"]);

  //// clear up
  cudaFree(cor.input_dptr());
  cudaFree(d_compressed_dump);
  cudaFree(d_decompressed);

  cudaStreamDestroy(stream);
  delete[] cor.input_hptr();
}