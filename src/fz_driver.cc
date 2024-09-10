
#include "fz_driver.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fz_module.hh"
#include "fz_utils.hh"
#include "utils/io.hh"

namespace {

static void check_cuda_error(cudaError_t status, const char* file, int line)
{
  if (cudaSuccess != status) {
    printf("\n");
    printf(
        "CUDA API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n",  //
        file, line, cudaGetErrorString(status), status);
    exit(EXIT_FAILURE);
  }
}

}  // namespace

#define CHECK_CUDA2(err) (check_cuda_error(err, __FILE__, __LINE__))

namespace fzgpu {

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
  CHECK_CUDA2(cudaMalloc(&d_decomp_output, sizeof(float) * pad_len));

  CHECK_CUDA2(cudaMalloc(&d_offset_counter, sizeof(uint32_t)));
  CHECK_CUDA2(cudaMalloc(&d_start_pos, sizeof(uint32_t) * grid_x));
  CHECK_CUDA2(cudaMalloc(&d_comp_size, sizeof(uint32_t) * grid_x));

  CHECK_CUDA2(cudaMemset(d_quantcode, 0, sizeof(uint16_t) * pad_len));
  CHECK_CUDA2(cudaMemset(d_bitflag_array, 0, sizeof(uint32_t) * chunk_size));
  CHECK_CUDA2(cudaMemset(d_decomp_quantcode, 0, sizeof(uint16_t) * pad_len));

  CHECK_CUDA2(cudaMemset(d_offset_counter, 0, sizeof(uint32_t)));
  CHECK_CUDA2(cudaMemset(d_start_pos, 0, sizeof(uint32_t) * grid_x));
  CHECK_CUDA2(cudaMemset(d_comp_size, 0, sizeof(uint32_t) * grid_x));
  CHECK_CUDA2(cudaMemset(d_decomp_output, 0, sizeof(float) * pad_len));

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
  CHECK_CUDA2(cudaFree(d_decomp_output));

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

size_t Compressor::postenc_calc_compressed_size()
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

void Compressor::decompress(uint16_t* d_archive, double const eb, void* stream)
{
  float _time;
  decomp_start = std::chrono::system_clock::now();

  fzgpu::cuhip::GPU_FZ_decode(
      d_archive, buf->d_decomp_quantcode, buf->d_bitflag_array,
      buf->d_start_pos, x, y, z, (cudaStream_t)stream);

  fzgpu::cuhip::GPU_reverse_lorenzo_predict_fz_variant(
      buf->d_signum, buf->d_decomp_quantcode, buf->d_decomp_output, len3, eb,
      _time, (cudaStream_t)stream);

  cudaStreamSynchronize((cudaStream_t)stream);
  decomp_end = std::chrono::system_clock::now();
}

void fzgpu_cli_compressor_roundtrip(
    std::string fname, int const x, int const y, int const z, double eb)
{
}

// TODO move to example/test
void fzgpu_reference_compressor_roundtrip(
    std::string fname, int const x, int const y, int const z, double eb)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  auto buf = fzgpu::internal_membuf(config);

  auto len3 = dim3(x, y, z);

  float time_elapsed;

  fzgpu::time_t comp_start, comp_end, decomp_start, decomp_end;

  buf.h_input = io::read_binary_to_new_array<float>(fname, config["pad_len"]);
  CHECK_CUDA2(cudaMalloc(&buf.d_input, sizeof(float) * config["len"]));
  CHECK_CUDA2(cudaMemset(buf.d_input, 0x0, sizeof(float) * config["len"]));

  double range =
      *std::max_element(buf.h_input, buf.h_input + config["pad_len"]) -
      *std::min_element(buf.h_input, buf.h_input + config["pad_len"]);

  eb *= range;

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
      buf.d_signum, buf.d_decomp_quantcode, buf.d_decomp_output, len3, eb,
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
      buf.h_decomp_output, buf.d_decomp_output, sizeof(float) * config["len"],
      cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  bool prequant_verify = true;
  if (bitshuffle_verify)
    prequant_verify = fzgpu::utils::prequantization_verify(
        buf.h_input, buf.h_decomp_output, config["len"], eb);

  fzgpu::utils::verify_data<float>(
      buf.h_decomp_output, buf.h_input, config["len"]);

  // print verification result
  if (bitshuffle_verify) {
    printf("bitshuffle veri successful!\n");
    if (prequant_verify)
      printf("prequant veri successful!\n");
    else
      printf("prequant veri fail\n");
  }
  else {
    printf("bitshuffle veri fail\n");
  }

  ////////////////////////////////////////////////////////////////////////////////

  CHECK_CUDA2(cudaMemcpy(
      &buf.h_offset_sum, buf.d_offset_counter, sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  printf("original size: %ld\n", config["bytes"]);
  auto comp_size = sizeof(uint32_t) * config["chunk_size"] +
                   buf.h_offset_sum * sizeof(uint32_t) +
                   sizeof(uint32_t) * int(config["quantcode_bytes"] / 4096);
  printf("comp_size: %ld\n", comp_size);
  printf("comp_ratio: %f\n", float(config["bytes"]) / float(comp_size));

  duration_t comp_time = comp_end - comp_start;
  duration_t decomp_time = decomp_end - decomp_start;

  fzgpu::utils::print_speed(
      comp_time.count(), decomp_time.count(), config["bytes"]);

  ////////////////////////////////////////////////////////////////////////////////

  cudaStreamDestroy(stream);

  delete[] buf.h_input;

  return;
}

void fzgpu_compressor_roundtrip_v1(
    std::string fname, int const x, int const y, int const z, double eb)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  Compressor cor(x, y, z);
  double range;

  cor.input_hptr() =
      io::read_binary_to_new_array<float>(fname, config["pad_len"]);
  CHECK_CUDA2(cudaMalloc(&cor.input_dptr(), sizeof(float) * config["len"]));
  CHECK_CUDA2(cudaMemset(cor.input_dptr(), 0, sizeof(float) * config["len"]));

  Compressor::profile_data_range(cor.input_hptr(), config["pad_len"], range);
  eb *= range;

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

  //// copy out
  uint16_t* d_compressed_dump;
  CHECK_CUDA2(
      cudaMalloc(&d_compressed_dump, compressed_size * sizeof(uint16_t)));
  CHECK_CUDA2(cudaMemcpy(
      d_compressed_dump, d_compressed_internal,
      sizeof(uint16_t) * compressed_size, cudaMemcpyDeviceToDevice));

  //// decompress using external saved archive
  cor.decompress(d_compressed_dump, eb, stream);

  CHECK_CUDA2(cudaMemcpy(
      cor.membuf()->h_quantcode, cor.membuf()->d_quantcode,
      sizeof(uint16_t) * config["len"], cudaMemcpyDeviceToHost));

  // bitshuffle verification
  CHECK_CUDA2(cudaMemcpy(
      cor.membuf()->h_decomp_quantcode, cor.membuf()->d_decomp_quantcode,
      sizeof(uint16_t) * config["len"], cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  auto bitshuffle_verify = fzgpu::utils::bitshuffle_verify(
      cor.membuf()->h_quantcode, cor.membuf()->h_decomp_quantcode,
      config["len"]);

  // prequant verification
  CHECK_CUDA2(cudaMemcpy(
      cor.membuf()->h_decomp_output, cor.membuf()->d_decomp_output,
      sizeof(float) * config["len"], cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  bool prequant_verify = true;
  if (bitshuffle_verify)
    prequant_verify = fzgpu::utils::prequantization_verify(
        cor.membuf()->h_input, cor.membuf()->h_decomp_output, config["len"],
        eb);

  fzgpu::utils::verify_data<float>(
      cor.membuf()->h_decomp_output, cor.membuf()->h_input, config["len"]);

  // print verification result
  if (bitshuffle_verify) {
    printf("bitshuffle veri successful!\n");
    if (prequant_verify)
      printf("prequant veri successful!\n");
    else
      printf("prequant veri fail\n");
  }
  else {
    printf("bitshuffle veri fail\n");
  }

  ////////////////////////////////////////////////////////////////////////////////

  CHECK_CUDA2(cudaMemcpy(
      &cor.membuf()->h_offset_sum, cor.membuf()->d_offset_counter,
      sizeof(uint32_t), cudaMemcpyDeviceToHost));

  printf("original size: %ld\n", config["bytes"]);
  auto comp_size =
      sizeof(uint32_t) * config["chunk_size"] +        // d_bitflag_array
      cor.membuf()->h_offset_sum * sizeof(uint32_t) +  // partially d_comp_out
      sizeof(uint32_t) * int(config["quantcode_bytes"] / 4096);  // start_pos
  printf("comp_size: %ld\n", comp_size);
  printf("comp_ratio: %f\n", float(config["bytes"]) / float(comp_size));

  duration_t comp_time = cor.comp_end - cor.comp_start;
  duration_t decomp_time = cor.decomp_end - cor.decomp_start;

  fzgpu::utils::print_speed(
      comp_time.count(), decomp_time.count(), config["bytes"]);

  ////////////////////////////////////////////////////////////////////////////////

  cudaStreamDestroy(stream);

  delete[] cor.membuf()->h_input;

  return;
}

}  // namespace fzgpu

#undef CHECK_CUDA2
