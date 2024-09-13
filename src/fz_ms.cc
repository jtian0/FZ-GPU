
#include <cuda_runtime.h>
#include <omp.h>

#include <vector>

#include "fz_driver.hh"
#include "fz_module.hh"
#include "fz_utils.hh"
#include "utils/io.hh"

// utilities for demo
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

using T = float;
using copybuf_t = uint8_t*;
using compressor_t = fzgpu::Compressor*;

T *d_emu_ready_data, *h_emu_ready_data;

static const int repeat = 100;

std::string fname;
static const size_t len1_100snapshots = 384 * 352 * 1600;  // 100 snapshots
static const size_t bytes_100snapshots = sizeof(T) * len1_100snapshots;
static const size_t super_cyclic = 1600;

static const int x = 384, y = 352;
static const int super_z = 1600;

static const size_t len1_4 = 384 * 352 * 4;
static const int z_4 = 4;
static const int zshift_4 = 4;

static const size_t len1_8 = 384 * 352 * 8;
static const int z_8 = 8;
static const int zshift_8 = 8;

static const size_t selected_len1 = len1_8;
static const int selected_z = z_8;
static const int selected_zshift = zshift_8;

// auto mode = Abs;  // set compression mode
auto eb = 3;  // set error bound

void multistream_compress_v1(int const x, int const y, int const z)
{
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());

  auto streams = new cudaStream_t[omp_get_num_procs()];
  auto compressor_instances = new compressor_t[omp_get_num_procs()];
  auto copy_buffers = new copybuf_t[omp_get_num_procs()];
  auto ptr_compressed = new uint16_t*[omp_get_num_procs()];
  auto compressed_sizes = new size_t[omp_get_num_procs()];

  // initialize ready data
  {
    cudaMalloc(&d_emu_ready_data, bytes_100snapshots);
    cudaMallocHost(&h_emu_ready_data, bytes_100snapshots);
    io::read_binary_to_array(fname, h_emu_ready_data, len1_100snapshots);
    cudaMemcpy(
        d_emu_ready_data, h_emu_ready_data, bytes_100snapshots,
        cudaMemcpyHostToDevice);
  }

// initialize each thread
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto num_cpu_threads = omp_get_num_threads();

    printf("cpu_id: %d/%d\n", tid, num_cpu_threads);

    // clang-format off
    cudaStreamCreate(&streams[tid]);
    compressor_instances[tid] = new fzgpu::Compressor(x,y * selected_z, 1);
    // copy_buffers[tid] = new uint8_t[selected_len1 * 2];  // CR upperlim=2
    // clang-format on
  }

// data processing
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();

    uint8_t* p_compressed;
    size_t comp_len;
    void* comp_timerecord;

    // auto dummy_copy_len = 1 / 4 * selected_len1 * sizeof(float);

    for (auto i = 0; i < repeat; i++) {
      // auto this_tid_shift =
      //     i * (omp_get_num_procs() * selected_len1) + tid * selected_len1;
      // this_tid_shift %= (super_cyclic / 2);
      auto this_tid_shift = 0;

      compressor_instances[tid]->compress(
          d_emu_ready_data + this_tid_shift, eb, &ptr_compressed[tid],
          &compressed_sizes[tid], streams[tid]);
    }
    cudaStreamSynchronize(streams[tid]);
  }

// clean up
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    cudaStreamDestroy(streams[tid]);
    delete[] copy_buffers[tid];
  }
  delete[] streams;
  delete[] compressor_instances;
  delete[] ptr_compressed, delete[] compressed_sizes;
  delete[] copy_buffers;
}

void multistream_compress_v2()
{
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());

  auto num_cpu_threads = omp_get_num_procs();
  auto streams = new cudaStream_t[num_cpu_threads];
  auto copy_buffers = new copybuf_t[num_cpu_threads];
  auto ptr_compressed = new uint16_t*[num_cpu_threads];
  auto compressed_sizes = new size_t[num_cpu_threads];
  auto membuf = new fzgpu::internal_membuf*[num_cpu_threads];

  // initialize ready data
  {
    cout << "bytes_100snapshots: " << bytes_100snapshots << endl;
    cudaMalloc(&d_emu_ready_data, bytes_100snapshots);
    cudaMallocHost(&h_emu_ready_data, bytes_100snapshots);
    io::read_binary_to_array(fname, h_emu_ready_data, len1_100snapshots);
    cudaMemcpy(
        d_emu_ready_data, h_emu_ready_data, bytes_100snapshots,
        cudaMemcpyHostToDevice);
  }

  cout << x << endl;
  cout << y << endl;
  cout << selected_z << endl;

  // auto config = fzgpu::utils::configure_fzgpu(x, y, selected_z);
  fzgpu::config_map config;
  config["len"] = 1081344;
  config["bytes"] = 4325376;
  config["pad_len"] = 1081344;
  config["chunk_size"] = 4224;
  config["quantcode_bytes"] = 2162688;
  config["grid_x"] = 528;

// initialize each thread
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto num_cpu_threads = omp_get_num_threads();

    printf("cpu_id: %d/%d\n", tid, num_cpu_threads);

    cudaStreamCreate(&streams[tid]);
    // membuf[tid] =
    //     new fzgpu::internal_membuf(config);  // supposed array for streams
    membuf[tid] = new fzgpu::internal_membuf(config);  // testing
  }

// data processing
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto buf = membuf[tid];
    auto len3 = dim3(x, y, selected_z);

    for (auto i = 0; i < repeat; i++) {
      auto this_tid_shift = 0;
      float _time;

      fzgpu::cuhip::GPU_lorenzo_predict_fz_variant(
          d_emu_ready_data + 128 * tid, buf->d_quantcode, buf->d_signum, len3,
          eb, _time, streams[tid]);

      fzgpu::cuhip::GPU_FZ_encode(
          buf->d_quantcode, buf->d_comp_out, buf->d_offset_counter,
          buf->d_bitflag_array, buf->d_start_pos, buf->d_comp_size, x, y,
          selected_z, streams[tid]);

      cudaStreamSynchronize(streams[tid]);
    }

// testing
#pragma omp parallel
    {
      // d2h copy quantization code
    }

// clean up
#pragma omp parallel
    {
      auto tid = omp_get_thread_num();
      cudaStreamDestroy(streams[tid]);
      delete[] copy_buffers[tid];
      delete membuf[tid];
    }
    delete[] streams;
    // delete[] compressor_instances;
    delete[] ptr_compressed, delete[] compressed_sizes;
    delete[] copy_buffers;
    delete[] membuf;
  }
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    /* For demo, we use 3600x1800 CESM data. */
    printf("PROG /path/to/super-data\n");
    exit(0);
  }

  fname = std::string(argv[1]);

  // multistream_compress_v1(x, y * super_z, 1);
  multistream_compress_v2();

  return 0;
}