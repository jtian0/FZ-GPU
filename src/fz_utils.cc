#include "fz_utils.hh"

#include <cmath>
#include <iostream>

using std::cout;
using std::endl;

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

bool bitshuffle_verify(
    uint16_t const* const h_quantcode,
    uint16_t const* const h_decomp_quantcode, size_t const len)
{
  // printf("begin bitshuffle veri\n");
  bool bitshuffle_verify = true;
  for (auto i = 0; i < len; i++) {
    if (h_quantcode[i] != h_decomp_quantcode[i]) {
      printf("data type len: %lu\n", len);
      printf(
          "verification failed at index: %d\noriginal quantization code: "
          "%u\ndecompressed quantization code: %u\n",
          i, h_quantcode[i], h_decomp_quantcode[i]);
      bitshuffle_verify = false;
      break;
    }
  }
  return bitshuffle_verify;
}

bool prequantization_verify(
    float const* const h_input, float const* const h_decomp_output,
    size_t const len, double const eb)
{
  bool prequant_verify = true;

  // printf("begin pre-quant veri\n");
  for (int i = 0; i < len; i++) {
    if (std::abs(h_input[i] - h_decomp_output[i]) > float(eb * 1.01)) {
      printf(
          "verification failed at index: %d\noriginal data: "
          "%f\ndecompressed data: %f\n",
          i, h_input[i], h_decomp_output[i]);
      printf(
          "error is: %f, while error bound is: %f\n",
          std::abs(h_input[i] - h_decomp_output[i]), float(eb));
      prequant_verify = false;
      break;
    }
  }
  return prequant_verify;
}

void print_speed(
    double const comp_time, double const decomp_time, size_t const bytes)
{
  auto GiBps = [](auto bytes, auto time) {
    return float(bytes) / 1024 / 1024 / 1024 / time;
  };

  printf(
      "comp_e2e_time (us): %f\n"
      "comp_e2e_speed (GiB/s): %f\n"
      "decomp_e2e_time (us): %f\n"
      "decomp_e2e_speed (GiB/s): %f\n",
      comp_time * 1e6, GiBps(bytes, comp_time),  //
      decomp_time * 1e6, GiBps(bytes, comp_time));
}

}  // namespace fzgpu::utils

template void fzgpu::utils::verify_data(
    float* xdata, float* odata, size_t len);