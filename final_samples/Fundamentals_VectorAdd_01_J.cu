```cpp
/*
Add error checking to all CUDA calls and the kernel launch.
Thinking:
- The goal is to create a complete CUDA program that demonstrates
  usage of all CUDA API calls while ensuring each call is checked for
  errors.
- Use a macro `CUDA_CHECK` that wraps any CUDA call, captures its
  return value, and if the result is not `cudaSuccess`, prints a
  descriptive error message (including the call text, file name,
  line number, and the CUDA error string) and exits the program.
- Every CUDA runtime API call in the code (e.g., `cudaMalloc`,
  `cudaMemcpy`, `cudaFree`, `cudaDeviceSynchronize`) will be
  wrapped with this macro.
- For kernel launches, after launching the kernel, we will call
  `cudaGetLastError` to catch launch failures and
  `cudaDeviceSynchronize` to catch runtime errors that occur during
  execution. Both of these calls will also be wrapped by `CUDA_CHECK`.
- The example program will perform a simple vector addition. It
  allocates two input vectors and one output vector on the device,
  copies data from the host to the device, launches a kernel that
  adds the vectors element-wise, copies the result back, prints a
  few output values for verification, and finally frees all
  allocated memory.
- All operations are protected by error checking to satisfy the
  requirement of adding error checking to all CUDA calls and the
  kernel launch.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls */
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

/* Simple vector addition kernel */
__global__ void vectorAdd(const float* A, const float* B, float* C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 20;   // 1M elements
    const size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    /* Device pointers */
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for kernel to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Simple verification */
    printf("Verification:\n");
    for (size_t i = 0; i < 5; ++i) {
        printf("h_A[%zu] = %f, h_B[%zu] = %f, h_C[%zu] = %f\n",
               i, h_A[i], i, h_B[i], i, h_C[i]);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully.\n");

    return EXIT_SUCCESS;
}
```