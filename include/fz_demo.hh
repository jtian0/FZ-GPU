#ifndef D9115CBA_CFAC_4884_B2B2_34279EA38AC0
#define D9115CBA_CFAC_4884_B2B2_34279EA38AC0

#include <string>

namespace fzgpu {

void compressor_roundtrip_v0(
    std::string fname, int const x, int const y, int const z, double eb,
    bool use_rel);
void compressor_roundtrip_v1(
    std::string fname, int const x, int const y, int const z, double eb,
    bool use_rel);

}  // namespace fzgpu

#endif /* D9115CBA_CFAC_4884_B2B2_34279EA38AC0 */
