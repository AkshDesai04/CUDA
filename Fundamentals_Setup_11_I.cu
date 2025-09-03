/*
Add the `-arch=sm_XX` flag to your `nvcc` command line to compile for a specific compute capability (e.g., `sm_75`).

[Thinking]
The prompt asks to demonstrate the use of the `-arch=sm_XX` flag. The simplest way to illustrate this is to provide a small CUDA program that can be compiled with that flag. I will include a short vector addition kernel and a main function that allocates device memory, launches the kernel, and copies results back to the host. The comment at the top contains the exact phrase from the prompt and explains that this code is ready to be compiled with a compute capability flag such as `-arch=sm_75`. No additional explanation or text should be output beyond the .cu file content itself.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20;           // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f)
        {
            success = false;
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
            break;
        }
    }
    if (success)
        printf("Vector addition succeeded.\n");

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
