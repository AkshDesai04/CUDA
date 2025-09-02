/*
Implement `C[i] = pow(A[i], 2.5f)`.

Thinking process:
- The requirement is to compute the 2.5 power of each element in array A and store the result in array C.
- In CUDA, we will write a kernel that performs this operation element-wise.
- The powf function (float powf(float, float)) is available in device code for computing floating‑point powers.
- We'll allocate memory on the device for both arrays, copy input data from host to device, launch the kernel, copy the result back, and finally print the results on the host.
- For simplicity, we'll use a single dimensional array of size N (e.g., N=10) and initialize A with some sample values.
- The kernel will use the usual block/grid indexing pattern: index = blockIdx.x * blockDim.x + threadIdx.x.
- We'll include basic CUDA error checking to aid debugging.
- Finally, the program is a self‑contained .cu file that can be compiled with nvcc and executed on a CUDA capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel: compute C[i] = powf(A[i], 2.5f)
__global__ void powKernel(const float* A, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = powf(A[idx], 2.5f);
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main()
{
    const int N = 10;
    size_t bytes = N * sizeof(float);

    // Host arrays
    float h_A[N];
    float h_C[N];

    // Initialize host input array with some values
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)(i + 1); // 1.0, 2.0, ..., 10.0
    }

    // Device arrays
    float *d_A = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input A:\n");
    for (int i = 0; i < N; ++i)
        printf("  A[%d] = %f\n", i, h_A[i]);

    printf("\nOutput C (A[i] ^ 2.5):\n");
    for (int i = 0; i < N; ++i)
        printf("  C[%d] = %f\n", i, h_C[i]);

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
