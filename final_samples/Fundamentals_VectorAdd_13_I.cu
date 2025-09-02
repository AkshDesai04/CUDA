/*
Aim of the program: Verify the results on the CPU.

Thinking:
To demonstrate GPU computation and confirm its correctness, I decided to implement a simple vector addition kernel.
The program will:
1. Allocate and initialize two input vectors on the host.
2. Allocate memory on the device and copy the inputs to it.
3. Launch a CUDA kernel that performs element‑wise addition.
4. Copy the result back to the host.
5. Compute the same addition on the CPU.
6. Compare the GPU and CPU results, reporting any mismatches or success.
This approach verifies the GPU results against a trusted CPU implementation while illustrating basic CUDA memory handling and kernel launch syntax.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.", \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// CUDA kernel for vector addition
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1.0f; // arbitrary values
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    for (int i = 0; i < N; ++i)
    {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-5f)
        {
            mismatches++;
            if (mismatches <= 10) // print first few mismatches
            {
                printf("Mismatch at index %d: GPU=%f, CPU=%f\n",
                       i, h_C_gpu[i], h_C_cpu[i]);
            }
        }
    }

    if (mismatches == 0)
    {
        printf("Verification successful: All results match.\n");
    }
    else
    {
        printf("Verification failed: %d mismatches found.\n", mismatches);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
