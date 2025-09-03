```cuda
/* 
Aim of the program:
Launch your grid-stride kernel with a number of blocks equal to the number of SMs on your GPU, 
and a reasonable block size (e.g. 128 or 256). This is a common performance heuristic.

Thought process:
1. We want to demonstrate a simple CUDA kernel that uses a grid‑stride loop so that each thread can 
   process multiple elements when the problem size exceeds the number of threads launched.
2. To follow the heuristic, we query the device properties to obtain the number of streaming 
   multiprocessors (SMs). Each SM will get one block.
3. We choose a block size of 256 threads (a common choice that works well on most GPUs). 
4. The kernel will perform element‑wise addition of two input arrays into an output array.
5. Host code will allocate device memory, copy data, launch the kernel with the chosen grid and 
   block dimensions, and copy the result back for verification.
6. Basic CUDA error checking is included to make debugging easier. 
7. The code is fully self‑contained and can be compiled with nvcc and executed on any 
   CUDA‑capable device.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Kernel that performs element‑wise addition using a grid‑stride loop
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x; // Number of threads in the grid

    // Grid‑stride loop
    for (size_t i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t err_ = (err);                       \
        if (err_ != cudaSuccess) {                     \
            fprintf(stderr, "CUDA error %s:%d: %s\n",  \
                    __FILE__, __LINE__,                \
                    cudaGetErrorString(err_));         \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

int main()
{
    // Problem size
    const size_t N = 1 << 24; // About 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize input arrays
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Query device properties to get number of SMs
    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
    int smCount = devProp.multiProcessorCount;
    printf("Device has %d SMs.\n", smCount);

    // Kernel launch configuration
    const int blockSize = 256;          // Threads per block
    const int gridSize = smCount;       // One block per SM

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification of first few elements
    bool ok = true;
    for (size_t i = 0; i < 10; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            ok = false;
            printf("Mismatch at %zu: host %f, device %f\n", i, expected, h_C[i]);
            break;
        }
    }
    if (ok)
        printf("Result verification passed.\n");
    else
        printf("Result verification failed.\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```