#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

using std::string;

using std::cout;
using std::endl;

std::string data_no = "025";

#include "fz_module.hh"
#include "fz_utils.hh"
#include "utils/io.hh"

using fzgpu_time_t = std::chrono::time_point<std::chrono::system_clock>;
using fzgpu_duration_t = std::chrono::duration<double>;

using T = float;
using E = uint16_t;
using S = bool;
using M = uint32_t;
static const int TEST_REPEAT = 1000;

std::string fname;
static const int x = 384, y = 352;
int super_z = 400;  // 25 snapshots
static const size_t super_len1 = x * y * super_z;
static const size_t super_bytes = sizeof(T) * super_len1;

static const int z_8 = 8;
static const size_t len1_8 = x * y * z_8;

int selected_z = z_8;
size_t selected_len1 = x * y * selected_z;

static const float eb = 3.0;  // set error bound

int main(int argc, char **argv)
{
  if (argc == 3) {
    data_no = string(argv[1]);
    super_z = atoi(argv[1]) * 16;

    selected_z = atoi(argv[2]);
    selected_len1 = x * y * selected_z;
  }
  printf("data no:\t%s\n", data_no.c_str());
  printf("super z:\t%d\n", super_z);
  printf("selected z:\t%d\n", selected_z);
  printf("selected len1:\t%zu\n", selected_len1);

  auto STREAM_COUNT = omp_get_max_threads();

  // printf("");

  fzgpu_time_t comp_start, comp_end;

  T *in, *d_in;
  E *d_quant;
  S *d_signum;

  M *d_offset_counter;
  M *d_bitflag_array;
  M *d_start_pos;
  M *d_comp_size;
  uint16_t *d_comp_out;

  in = (T *)malloc(super_len1 * sizeof(T));  // input (host)

  cudaMalloc(&d_in, super_len1 * sizeof(T));
  cudaMemset(d_in, 0, super_len1 * sizeof(T));

  cudaMalloc(&d_quant, super_len1 * sizeof(E));
  cudaMemset(d_quant, 0, super_len1 * sizeof(E));

  cudaMalloc(&d_signum, super_len1 * sizeof(S));
  cudaMemset(d_signum, 0, super_len1 * sizeof(S));

  cudaMalloc(&d_offset_counter, super_len1 * sizeof(M));
  cudaMemset(d_offset_counter, 0, super_len1 * sizeof(M));

  cudaMalloc(&d_bitflag_array, super_len1 * sizeof(M));
  cudaMemset(d_bitflag_array, 0, super_len1 * sizeof(M));

  cudaMalloc(&d_start_pos, super_len1 * sizeof(M));
  cudaMemset(d_start_pos, 0, super_len1 * sizeof(M));

  cudaMalloc(&d_comp_size, super_len1 * sizeof(M) / 4096 + 1);
  cudaMemset(d_comp_size, 0, super_len1 * sizeof(M) / 4096 + 1);

  cudaMalloc(&d_comp_out, super_len1 * sizeof(M));
  cudaMemset(d_comp_out, 0, super_len1 * sizeof(M));

  // load data from disk (emulate data-readying before compression)
  io::read_binary_to_array(
      "/mnt/d/linux_data_overvlow/DO_NOT_SHARE/first_" + data_no + ".raw", in,
      super_len1);
  // H2D copy
  cudaMemcpy(&d_in, &in, super_bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cudaStream_t streams[STREAM_COUNT];

  cudaEvent_t start[STREAM_COUNT], stop[STREAM_COUNT];
  float milliseconds[STREAM_COUNT];

  bool graph_created[STREAM_COUNT];
  for (auto i = 0; i < STREAM_COUNT; i++) graph_created[i] = false;

  cudaGraph_t graphs[STREAM_COUNT];
  cudaGraphExec_t graph_instances[STREAM_COUNT];

  //// OMP zone
#pragma omp parallel
  {
    // TODO later on, we move stream creation in

#pragma omp for
    for (int i = 0; i < STREAM_COUNT; i++) {
      cudaStreamCreate(&streams[i]);
      cudaEventCreate(&start[i]);
      cudaEventCreate(&stop[i]);
      cudaEventRecord(start[i], streams[i]);
    }

#pragma omp master
    {
      comp_start = std::chrono::system_clock::now();
    }

#pragma omp for nowait
    for (auto i = 0; i < STREAM_COUNT; i++) {
      int offset = i * selected_len1;
      T placeholder_time;

      for (auto _ = 0; _ < TEST_REPEAT; _++) {
        //
        if (not graph_created[i]) {
          cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal);

          fzgpu::cuhip::GPU_lorenzo_predict_fz_variant(
              d_in + offset, d_quant + offset, d_signum + offset,
              dim3(x, y, selected_z), eb, placeholder_time, streams[i]);
          fzgpu::cuhip::GPU_FZ_encode(
              d_quant + offset, d_comp_out + offset, d_offset_counter + offset,
              d_bitflag_array + offset, d_start_pos + offset,
              d_comp_size + offset, x, y, selected_z, streams[i]);
          cudaStreamEndCapture(streams[i], &graphs[i]);
          cudaGraphInstantiate(&graph_instances[i], graphs[i], NULL, NULL, 0);
          graph_created[i] = true;
        }

        cudaGraphLaunch(graph_instances[i], streams[i]);
        // cudaStreamSynchronize(streams[i]);

        //
      }
      // end of repeat testing
    }
#pragma omp for
    for (auto i = 0; i < STREAM_COUNT; i++) {
      cudaStreamSynchronize(streams[i]);
      cudaEventRecord(stop[i], streams[i]);
      cudaEventSynchronize(stop[i]);
    }

#pragma omp master
    {
      //   cudaDeviceSynchronize();
      comp_end = std::chrono::system_clock::now();
      fzgpu_duration_t comp_time = comp_end - comp_start;
      printf("omp zone time (us)\t%.2lf\n", comp_time.count() * 1e6);
      printf("omp zone time (ms)\t%.2lf\n", comp_time.count() * 1e3);

      for (auto i = 0; i < STREAM_COUNT; i++) {
        cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
        printf("stream %02u cuda time (ms):\t%.2f\n", i, milliseconds[i]);
      }
    }
  }
  // We are not out of omp zone.

  //// OMP zone end
  ///////////////////////////////////////////////////////

  // clean up
  free(in);
  cudaFree(d_in), cudaFree(d_quant), cudaFree(d_signum);
  cudaFree(d_offset_counter), cudaFree(d_bitflag_array), cudaFree(d_start_pos);
  cudaFree(d_comp_size), cudaFree(d_comp_out);

  for (int i = 0; i < STREAM_COUNT; i++) { cudaStreamDestroy(streams[i]); }

  printf("\n---------------------------\n");
  printf("whole size (not bytes) = %ld\n", super_len1);
  printf("number of streams used = %d\n", STREAM_COUNT);
  printf("\n");

  return 0;
}