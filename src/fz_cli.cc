#include <string>
#include <iostream>

using std::cout;
using std::endl;

#include "fz_driver.hh"

int main(int argc, char* argv[])
{
  if (argc < 6) {
    printf("    1      2  3  4  5\n");
    printf("fz  fname  x  y  z  rel-eb\n");
    exit(1);
  }

  auto const fname = std::string(argv[1]);
  auto const x = std::stoi(argv[2]);
  auto const y = std::stoi(argv[3]);
  auto const z = std::stoi(argv[4]);
  auto const eb = std::stod(argv[5]);

  cout << "\nfzgpu_reference_compressor_roundtrip\n" << endl;
  fzgpu::fzgpu_reference_compressor_roundtrip(fname, x, y, z, eb);
  cout << "\nfzgpu_compressor_roundtrip_v1\n" << endl;
  fzgpu::fzgpu_compressor_roundtrip_v1(fname, x, y, z, eb);
  return 0;
}