/*
Aim of the program: Verify the results on the CPU.

Thinking:
I will write a simple CUDA program that performs element-wise addition of two vectors on the GPU.
After the GPU computation, the result will be copied back to the host.
I will then compute the same addition on the CPU and compare each element to confirm that the GPU results are correct.
The code includes:
- Basic CUDA error checking.
- Memory allocation on both host and device.
- A simple CUDA kernel for vector addition.
- CPU-side verification loop that prints a message if any mismatch occurs.
- Clean up of allocated memory.
This demonstrates how to validate GPU results against a CPU reference.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                 \
        {                                                                        \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    }

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU reference calculation
    for (int i = 0; i < N; ++i)
    {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (h_C_gpu[i] != h_C_cpu[i])
        {
            if (errors < 10) // print first few errors
            {
                printf("Mismatch at index %d: GPU %f != CPU %f\n", i, h_C_gpu[i], h_C_cpu[i]);
            }
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("All results match. Verification succeeded.\n");
    }
    else
    {
        printf("Total mismatches: %d\n", errors);
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
