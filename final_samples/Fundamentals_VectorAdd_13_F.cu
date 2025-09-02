```cuda
/*
Aim of the program: Handle large vectors with boundary checks.

Thinking:
- The goal is to demonstrate how to process very large vectors on the GPU while ensuring that each thread accesses only valid memory locations.
- A common pattern for vector operations in CUDA is to launch a 1‑D grid of threads, where each thread is responsible for one element.
- To handle vectors that may not be a perfect multiple of the block size, the kernel must perform a boundary check (`if (idx < N)`) before reading or writing.
- We will implement a simple element‑wise addition: C = A + B, where A, B, and C are large arrays of floats.
- The host code will:
  1. Allocate host memory for two input vectors and one output vector.
  2. Initialize the input vectors with some values (e.g., i and 2*i).
  3. Allocate device memory and copy the inputs to the device.
  4. Launch the kernel with an appropriate block and grid size.
  5. Copy the result back to the host.
  6. Verify a few elements to ensure correctness.
  7. Clean up all allocated memory.
- Error checking is performed after every CUDA API call using a helper macro to make the code robust.
- The program can handle vectors larger than the number of available GPU threads by simply launching enough blocks; the boundary check guarantees safety.
- For demonstration, a vector size of 1 << 24 (~16 million) is used, which is typical for testing large data handling.
- The code is written in C with CUDA extensions (.cu file) and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel for element-wise addition: C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 24; // 16 million elements
    const size_t SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(SIZE);
    float *h_B = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input vectors
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_B, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_C, SIZE));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 5; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected)
        {
            fprintf(stderr, "Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_C[i], expected);
            ++errors;
        }
    }
    if (errors == 0)
    {
        printf("Verification passed for first 5 elements.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```