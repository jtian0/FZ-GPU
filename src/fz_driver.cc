#include "fz_driver.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fz_module.hh"
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

namespace fzgpu::utils {

template <typename T>
void verify_data(T* xdata, T* odata, size_t len)
{
  double max_odata = odata[0], min_odata = odata[0];
  double max_xdata = xdata[0], min_xdata = xdata[0];
  double max_abserr = max_abserr = fabs(xdata[0] - odata[0]);

  double sum_0 = 0, sum_x = 0;
  for (size_t i = 0; i < len; i++) sum_0 += odata[i], sum_x += xdata[i];

  double mean_odata = sum_0 / len, mean_xdata = sum_x / len;
  double sum_var_odata = 0, sum_var_xdata = 0, sum_err2 = 0, sum_corr = 0,
         rel_abserr = 0;

  double max_pwrrel_abserr = 0;
  size_t max_abserr_index = 0;
  for (size_t i = 0; i < len; i++) {
    max_odata = max_odata < odata[i] ? odata[i] : max_odata;
    min_odata = min_odata > odata[i] ? odata[i] : min_odata;

    max_xdata = max_xdata < odata[i] ? odata[i] : max_xdata;
    min_xdata = min_xdata > xdata[i] ? xdata[i] : min_xdata;

    float abserr = fabs(xdata[i] - odata[i]);
    if (odata[i] != 0) {
      rel_abserr = abserr / fabs(odata[i]);
      max_pwrrel_abserr =
          max_pwrrel_abserr < rel_abserr ? rel_abserr : max_pwrrel_abserr;
    }
    max_abserr_index = max_abserr < abserr ? i : max_abserr_index;
    max_abserr = max_abserr < abserr ? abserr : max_abserr;
    sum_corr += (odata[i] - mean_odata) * (xdata[i] - mean_xdata);
    sum_var_odata += (odata[i] - mean_odata) * (odata[i] - mean_odata);
    sum_var_xdata += (xdata[i] - mean_xdata) * (xdata[i] - mean_xdata);
    sum_err2 += abserr * abserr;
  }
  double std_odata = sqrt(sum_var_odata / len);
  double std_xdata = sqrt(sum_var_xdata / len);
  double ee = sum_corr / len;

  double inputRange = max_odata - min_odata;
  double mse = sum_err2 / len;
  double psnr = 20 * log10(inputRange) - 10 * log10(mse);
  std::cout << "PSNR: " << psnr << std::endl;
}

}  // namespace fzgpu::utils

#define CHECK_CUDA2(err) (check_cuda_error(err, __FILE__, __LINE__))

namespace fzgpu {

internal_membuf::internal_membuf(config_map config, bool _verifiy_on) :
    verify_on(_verifiy_on)
{
  auto grid_x = config["grid_x"];
  auto pad_len = config["pad_len"];
  auto chunk_size = config["chunk_size"];

  CHECK_CUDA2(cudaMalloc(&d_input, sizeof(float) * pad_len));
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
  CHECK_CUDA2(cudaFree(d_quantcode));
  CHECK_CUDA2(cudaFree(d_input));
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

void fzgpu_compressor_roundtrip(
    std::string fname, int const x, int const y, int const z, double eb)
{
  auto config = fzgpu::utils::configure_fzgpu(x, y, z);
  auto buf = fzgpu::internal_membuf(config);

  auto len3 = dim3(x, y, z);

  float time_elapsed;

  fzgpu::time_t comp_start, comp_end, decomp_start, decomp_end;

  buf.h_input = io::read_binary_to_new_array<float>(fname, config["pad_len"]);
  double range =
      *std::max_element(buf.h_input, buf.h_input + config["pad_len"]) -
      *std::min_element(buf.h_input, buf.h_input + config["pad_len"]);

  CHECK_CUDA2(cudaMemcpy(
      buf.d_input, buf.h_input, sizeof(float) * config["len"],
      cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  comp_start = std::chrono::system_clock::now();

  fzgpu::cuhip::GPU_lorenzo_predict_fz_variant(
      buf.d_input, buf.d_quantcode, buf.d_signum, len3, eb * range,
      time_elapsed, stream);

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
      buf.d_signum, buf.d_decomp_quantcode, buf.d_decomp_output, len3,
      eb * range, time_elapsed, stream);

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

  printf("begin bitshuffle verification\n");
  bool bitshuffle_verify = true;
  for (int tmp_idx = 0; tmp_idx < config["len"]; tmp_idx++) {
    if (buf.h_quantcode[tmp_idx] != buf.h_decomp_quantcode[tmp_idx]) {
      printf("data type len: %lu\n", config["len"]);
      printf(
          "verification failed at index: %d\noriginal quantization code: "
          "%u\ndecompressed quantization code: %u\n",
          tmp_idx, buf.h_quantcode[tmp_idx], buf.h_decomp_quantcode[tmp_idx]);
      bitshuffle_verify = false;
      break;
    }
  }

  // pre-quantization verification
  CHECK_CUDA2(cudaMemcpy(
      buf.h_decomp_output, buf.d_decomp_output, sizeof(float) * config["len"],
      cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(stream);

  bool prequant_verify = true;
  if (bitshuffle_verify) {
    printf("begin pre-quantization verification\n");
    for (int tmp_idx = 0; tmp_idx < config["len"]; tmp_idx++) {
      if (std::abs(buf.h_input[tmp_idx] - buf.h_decomp_output[tmp_idx]) >
          float(eb * 1.01 * range)) {
        printf(
            "verification failed at index: %d\noriginal data: "
            "%f\ndecompressed data: %f\n",
            tmp_idx, buf.h_input[tmp_idx], buf.h_decomp_output[tmp_idx]);
        printf(
            "error is: %f, while error bound is: %f\n",
            std::abs(buf.h_input[tmp_idx] - buf.h_decomp_output[tmp_idx]),
            float(eb * range));
        prequant_verify = false;
        break;
      }
    }
  }

  fzgpu::utils::verify_data<float>(
      buf.h_decomp_output, buf.h_input, config["len"]);

  // print verification result
  if (bitshuffle_verify) {
    printf("bitshuffle verification succeed!\n");
    if (prequant_verify)
      printf("pre-quantization verification succeed!\n");
    else
      printf("pre-quantization verification fail\n");
  }
  else {
    printf("bitshuffle verification fail\n");
  }

  ////////////////////////////////////////////////////////////////////////////////

  CHECK_CUDA2(cudaMemcpy(
      &buf.h_offset_sum, buf.d_offset_counter, sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  printf("original size: %ld\n", config["bytes"]);
  printf(
      "compressed size: %ld\n",
      sizeof(uint32_t) * config["chunk_size"] +
          buf.h_offset_sum * sizeof(uint32_t) +
          sizeof(uint32_t) * int(config["quantcode_bytes"] / 4096));
  printf(
      "compression ratio: %f\n",
      float(config["bytes"]) /
          float(
              sizeof(uint32_t) * config["chunk_size"] +
              buf.h_offset_sum * sizeof(uint32_t) +
              sizeof(uint32_t) * floor(config["quantcode_bytes"] / 4096)));

  std::chrono::duration<double> comp_time = comp_end - comp_start;
  std::chrono::duration<double> decomp_time = decomp_end - decomp_start;

  std::cout << "compression e2e time: " << comp_time.count() << " s\n";
  std::cout << "compression e2e throughput: "
            << float(config["bytes"]) / 1024 / 1024 / 1024 / comp_time.count()
            << " GB/s\n";

  std::cout << "decompression e2e time: " << decomp_time.count() << " s\n";
  std::cout << "decompression e2e throughput: "
            << float(config["bytes"]) / 1024 / 1024 / 1024 /
                   decomp_time.count()
            << " GB/s\n";

  ////////////////////////////////////////////////////////////////////////////////

  cudaStreamDestroy(stream);

  delete[] buf.h_input;

  return;
}

}  // namespace fzgpu

#undef CHECK_CUDA2
