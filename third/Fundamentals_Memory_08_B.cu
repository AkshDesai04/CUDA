/*
Perform the same timing for a device-to-device copy.

The goal of this program is to benchmark the performance of a
device-to-device memory copy in CUDA.  The original context (not
provided here) likely involved timing host-to-device or device-to-host
copies, so we replicate the timing mechanism for a copy that occurs
entirely on the GPU.  The key steps are:

1. Allocate two device buffers of equal size.
2. Fill the source buffer with some data (copy from host or launch a
   kernel – here we use a host-to-device copy for simplicity).
3. Use CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`) to
   measure the elapsed time of multiple `cudaMemcpy` calls with the
   `cudaMemcpyDeviceToDevice` kind.
4. Report the average time per copy and the total time.
5. Clean up resources.

Important design choices:
- We use a 1 MB buffer and perform 1000 copies to get a measurable
  duration.
- Error checking is performed via a macro (`CUDA_CHECK`) that prints
  an error message and exits on failure.
- The timing is performed synchronously; we do not use streams or
  async copies because the focus is on the raw copy latency.
- A warm‑up copy is executed before the timed loop to avoid first‑time
  overhead skewing results.

This program is self‑contained, can be compiled with `nvcc`, and
outputs the timing results to the console.

*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for error checking
#define CUDA_CHECK(ans)                                              \
  {                                                                  \
    cudaError_t err = (ans);                                         \
    if (err != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error in file '%s' at line %d: %s\n",    \
              __FILE__, __LINE__, cudaGetErrorString(err));          \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  }

int main() {
  const size_t bytes = 1 << 20;      // 1 MB
  const int num_copies = 1000;

  // Allocate device memory
  float *d_src, *d_dst;
  CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_dst, bytes));

  // Allocate host memory and initialize
  float *h_tmp = (float*)malloc(bytes);
  for (size_t i = 0; i < bytes / sizeof(float); ++i) {
    h_tmp[i] = static_cast<float>(i) * 0.001f;
  }

  // Copy data from host to source device buffer
  CUDA_CHECK(cudaMemcpy(d_src, h_tmp, bytes, cudaMemcpyHostToDevice));

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warm-up copy
  CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));

  // Record start event
  CUDA_CHECK(cudaEventRecord(start, 0));

  // Perform device-to-device copies
  for (int i = 0; i < num_copies; ++i) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
  }

  // Record stop event
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Compute elapsed time
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Output results
  printf("Device-to-device copy:\n");
  printf("  Size per copy : %zu bytes\n", bytes);
  printf("  Number of copies : %d\n", num_copies);
  printf("  Total time : %f ms\n", ms);
  printf("  Avg time per copy : %f µs\n", (ms * 1000.0f) / num_copies);

  // Clean up
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  free(h_tmp);

  return 0;
}
