```cuda
/*
Aim of the program: Perform vector addition on GPU using double precision.

My entire thinking:
1. The user explicitly requested a CUDA program written in C, compiled as a .cu file, and specified that it should use double precision.
2. A common demonstration of double precision on CUDA is vector addition. It shows basic memory allocation, data transfer, kernel launch, and result verification.
3. I will implement the following components:
   - Define vector size (e.g., 1M elements) and allocate host memory.
   - Initialize host vectors with deterministic values for easy verification.
   - Allocate device memory for the vectors.
   - Copy input data from host to device.
   - Launch a CUDA kernel that adds corresponding elements of the two input vectors into a third vector.
   - Copy the result back to the host.
   - Verify the correctness by comparing with a CPU implementation.
4. For kernel launch configuration, I will use a grid of blocks where each block processes a chunk of the vector. The block size will be a typical value such as 256 threads per block, and the number of blocks will be computed to cover the entire vector length.
5. All memory allocations and transfers will include basic error checks using CUDA's error handling macros.
6. Since double precision arithmetic requires the GPU to support it, I will include a check to ensure the device has sufficient compute capability (>= 1.3 for double support). However, for brevity, the code will simply run assuming a capable device; advanced checks are omitted.
7. The code will be self-contained, compile with nvcc, and output a confirmation if the addition succeeded.

The resulting .cu file will follow standard CUDA conventions and can be compiled with:
   nvcc -arch=sm_61 -o vecadd_d vecadd_d.cu
or with an appropriate architecture flag for the target GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// Number of elements in each vector
#define N 1000000
// Threads per block
#define THREADS_PER_BLOCK 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Kernel for vector addition (C = A + B)
__global__ void vecAddKernel(const double *A, const double *B, double *C, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    double *h_A = NULL;
    double *h_B = NULL;
    double *h_C = NULL;
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;

    size_t size = N * sizeof(double);

    // Allocate host memory
    h_A = (double *)malloc(size);
    h_B = (double *)malloc(size);
    h_C = (double *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host vectors with deterministic values
    for (int i = 0; i < N; i++)
    {
        h_A[i] = (double)i * 1.0;
        h_B[i] = (double)(N - i) * 1.0;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy host input to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    vecAddKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++)
    {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9)
        {
            fprintf(stderr, "Result verification failed at element %d: GPU %f != CPU %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("Vector addition succeeded. All results match.\n");
    }
    else
    {
        printf("Vector addition failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```