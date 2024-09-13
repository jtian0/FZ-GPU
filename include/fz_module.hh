#ifndef A56D3730_71D1_4C49_A664_8028F190C0C0
#define A56D3730_71D1_4C49_A664_8028F190C0C0

// TODO document this:
// It is okay to only include PROJ_error_status at moduel level.
// However, it should not be inlined in this file.

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FZGPU_SUCCESS,
  FZGPU_GENERAL_GPU_FAILURE,  // translate from all cuda_errors
  FZGPU_NOT_IMPLEMENTED,
} fzgpu_error_status;

typedef fzgpu_error_status fzgpuerror;

#ifdef __cplusplus
}
#endif

#include <cuda_runtime.h>

#include <string>
#include <unordered_map>

#define NON_INTRUSIVE_MOD_2409 1

namespace fzgpu {

using config_map = std::unordered_map<std::string, size_t>;

}

namespace fzgpu::utils {

config_map configure_fzgpu(int const x, int const y, int const z);

}  // namespace fzgpu::utils

namespace fzgpu::cuhip {

using T = float;
using E = uint16_t;
using FP = float;

// TODO this is compat but redundant layer; remove the original
// launch_construct_LorenzoI_var and launch_reconstruct_LorenzoI_var later.
fzgpuerror GPU_lorenzo_predict_fz_variant(
    T* d_input, E* d_quantcode, bool* d_signum, dim3 const input_dim3,
    double const eb, float& time_elapsed, cudaStream_t stream);

fzgpuerror GPU_reverse_lorenzo_predict_fz_variant(
    bool* d_signum, E* d_quantcode, T* d_decomp_output, dim3 const input_dim3,
    double const eb, float& time_elapsed, cudaStream_t stream);

fzgpuerror GPU_FZ_encode(
    uint16_t* d_quantcode, uint16_t* d_comp_out, uint32_t* d_offset_counter,
    uint32_t* d_bitflag_array, uint32_t* d_start_pos, uint32_t* d_comp_size,
    int const x, int const y, int const z, cudaStream_t stream);

fzgpuerror GPU_FZ_decode(
    uint16_t* d_comp_out, uint16_t* d_decomp_quantcode,
    uint32_t* d_bitflag_array, uint32_t* d_start_pos, int const x, int const y,
    int const z, cudaStream_t stream);

}  // namespace fzgpu::cuhip

#endif /* A56D3730_71D1_4C49_A664_8028F190C0C0 */
