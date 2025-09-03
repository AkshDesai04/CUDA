```cuda
/* 
Aim of the program: Verify the results on the CPU.

Thinking:
To demonstrate how to validate GPU computations against CPU results, 
the program performs a simple element‑wise vector addition on the GPU
and then recomputes the same operation on the CPU. 
The program allocates two input vectors (A and B) and an output vector (C)
of a large size (e.g., 1,048,576 elements). The inputs are initialized
to deterministic values so that the CPU result is reproducible.
A CUDA kernel executes the addition on the device; the result is copied
back to host memory. The CPU then performs the same addition and the
two result arrays are compared element‑by‑element within a small
tolerance to account for floating‑point rounding. A mismatch count
is reported, and a success message is printed if no mismatches are
found. This example illustrates proper memory management, kernel
launch configuration, and correctness verification between host
and device computations. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N (1<<20)           // 1,048,576 elements
#define THREADS_PER_BLOCK 256
#define TOLERANCE 1e-5f

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Function to check for CUDA errors
void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    // Host memory allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C_gpu = (float*)malloc(N * sizeof(float));
    float *h_C_cpu = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = sinf(i) * 0.5f;
        h_B[i] = cosf(i) * 0.5f;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    checkCudaError("cudaMalloc d_A");
    cudaMalloc((void**)&d_B, N * sizeof(float));
    checkCudaError("cudaMalloc d_B");
    cudaMalloc((void**)&d_C, N * sizeof(float));
    checkCudaError("cudaMalloc d_C");

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy d_A");
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy d_B");

    // Determine grid dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    checkCudaError("kernel launch");

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy d_C to h_C_gpu");

    // CPU computation for verification
    for (int i = 0; i < N; ++i)
    {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; ++i)
    {
        float diff = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > TOLERANCE)
        {
            mismatches++;
            if (mismatches <= 10) // print first few mismatches
            {
                printf("Mismatch at index %d: GPU %f vs CPU %f (diff %f)\n",
                       i, h_C_gpu[i], h_C_cpu[i], diff);
            }
        }
    }

    if (mismatches == 0)
    {
        printf("Verification PASSED: All results match within tolerance.\n");
    }
    else
    {
        printf("Verification FAILED: %d mismatches found.\n", mismatches);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return EXIT_SUCCESS;
}
```