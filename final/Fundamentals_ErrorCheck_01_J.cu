/*
Aim: Add a conditional compilation flag (e.g., `_DEBUG`) so the `CHECK` macro only performs the check in debug builds, and is empty in release builds for performance.

Thinking: The goal is to provide a `CHECK` macro that performs CUDA error checking when we compile in debug mode, but does nothing (or only calls the function) in release mode to avoid the overhead of error handling. We use a preprocessor flag `_DEBUG` which is commonly defined by build systems when compiling debug versions. In debug builds, `CHECK` will execute the CUDA call, inspect the returned `cudaError_t`, and if it is not `cudaSuccess`, it will print an error message with file and line information and terminate the program. In release builds, we want to avoid the cost of error checking, so `CHECK` will simply invoke the CUDA call without inspecting the return value. The macro will be defined using a `do { ... } while(0)` idiom so it can be safely used like a normal function call. A small example kernel and a simple main program will demonstrate the macro usage. The program can be compiled with `nvcc -D_DEBUG test.cu` to enable the debug checks, or without `-D_DEBUG` to compile a release version where `CHECK` does not perform error checking.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Conditional CHECK macro definition */
#ifdef _DEBUG
/* In debug builds, perform error checking after each CUDA call */
#define CHECK(call)                                                     \
  do {                                                                  \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
              __FILE__, __LINE__, cudaGetErrorString(err));           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)
#else
/* In release builds, only invoke the call; no error checking */
#define CHECK(call) do { call; } while (0)
#endif

/* Simple vector addition kernel */
__global__ void addVectors(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
  const int N = 1 << 20; // 1M elements
  size_t size = N * sizeof(float);

  /* Allocate host memory */
  float *h_a = (float*)malloc(size);
  float *h_b = (float*)malloc(size);
  float *h_c = (float*)malloc(size);

  /* Initialize host arrays */
  for (int i = 0; i < N; ++i) {
    h_a[i] = (float)i;
    h_b[i] = (float)(N - i);
  }

  /* Allocate device memory */
  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc((void**)&d_a, size));
  CHECK(cudaMalloc((void**)&d_b, size));
  CHECK(cudaMalloc((void**)&d_c, size));

  /* Copy data from host to device */
  CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  /* Launch kernel */
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  /* Check kernel launch */
  CHECK(cudaGetLastError());
  /* Wait for GPU to finish */
  CHECK(cudaDeviceSynchronize());

  /* Copy result back to host */
  CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  /* Simple verification of a few elements */
  for (int i = 0; i < 5; ++i) {
    printf("c[%d] = %f\n", i, h_c[i]);
  }

  /* Clean up */
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
