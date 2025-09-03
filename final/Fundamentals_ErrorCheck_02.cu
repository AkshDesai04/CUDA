/*
Refactor your vector addition program from the previous section to use the `CHECK()` macro for every single CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, etc.).

Thinking:
- The aim is to demonstrate error checking on every CUDA runtime API call.
- We'll define a macro `CHECK(call)` that executes the CUDA call, checks the returned `cudaError_t`, prints an informative message on failure, and exits.
- All device memory allocations (`cudaMalloc`), memory copies (`cudaMemcpy`), and deallocations (`cudaFree`) will be wrapped by `CHECK`.
- After launching the kernel, we will check for launch errors using `cudaGetLastError()` and ensure synchronization with `cudaDeviceSynchronize()`, both wrapped by `CHECK`.
- The kernel itself remains unchanged; it's a simple element‑wise addition.
- The host code will allocate host arrays, initialize them, perform the device operations, and validate the result.
- No other supporting files are needed; everything is self‑contained in this .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                \
  {                                                                \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
      fprintf(stderr, "CUDA error in %s:%d : %s\n",                \
              __FILE__, __LINE__, cudaGetErrorString(err));        \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  }

/* Kernel: Element‑wise addition of vectors A and B into vector C */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

int main(void) {
  const int N = 1 << 20;          // 1M elements
  const int size = N * sizeof(float);

  /* Allocate host memory */
  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);
  if (!h_A || !h_B || !h_C) {
    fprintf(stderr, "Failed to allocate host vectors.\n");
    return EXIT_FAILURE;
  }

  /* Initialize host vectors */
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(2 * i);
  }

  /* Allocate device memory */
  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  CHECK(cudaMalloc((void**)&d_A, size));
  CHECK(cudaMalloc((void**)&d_B, size));
  CHECK(cudaMalloc((void**)&d_C, size));

  /* Copy host vectors to device */
  CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  /* Launch kernel */
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  /* Check for kernel launch errors */
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  /* Copy result back to host */
  CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  /* Simple verification */
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float expected = h_A[i] + h_B[i];
    if (fabs(h_C[i] - expected) > 1e-5f) {
      if (errors < 10) {
        fprintf(stderr, "Mismatch at index %d: GPU %f != CPU %f\n",
                i, h_C[i], expected);
      }
      ++errors;
    }
  }
  if (errors == 0) {
    printf("Vector addition succeeded, no errors found.\n");
  } else {
    printf("Vector addition completed with %d mismatches.\n", errors);
  }

  /* Free device memory */
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));

  /* Free host memory */
  free(h_A);
  free(h_B);
  free(h_C);

  return EXIT_SUCCESS;
}
