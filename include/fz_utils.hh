#ifndef F70E8BE5_845C_4BA8_B443_1D0FEA117DD5
#define F70E8BE5_845C_4BA8_B443_1D0FEA117DD5

#include <cstddef>
#include <cstdint>

namespace fzgpu::utils {

template <typename T>
void verify_data(T* xdata, T* odata, size_t len);

bool bitshuffle_verify(
    uint16_t const* const h_quantcode,
    uint16_t const* const h_decomp_quantcode, size_t const len);

bool prequantization_verify(
    float const* const h_input, float const* const h_decomp_output,
    size_t const len, double const eb);

void print_speed(
    double const comp_time, double const decomp_time, size_t const bytes);

}  // namespace fzgpu::utils

#endif /* F70E8BE5_845C_4BA8_B443_1D0FEA117DD5 */
