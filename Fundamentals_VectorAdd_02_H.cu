```cuda
/*
Pass `N` as an argument to the kernel.

Thinking:
The task is to demonstrate how to pass a variable N as a kernel argument in CUDA C. 
To keep the example clear, I will implement a simple element-wise operation on an array: 
each element of the input array is multiplied by 2 and stored in an output array. 
The size of the array, N, is supplied by the host program and passed directly to the kernel 
as a kernel argument. This shows that kernel arguments can be regular scalar values. 
The program includes:
1. Allocation of host input and output arrays.
2. Allocation of device input and output arrays.
3. Copy of input data to device.
4. Kernel launch with N passed in.
5. Copy of result back to host.
6. Verification of the result.
The kernel will use a standard 1D grid of 1D blocks; the number of threads per block 
is set to 256. The kernel checks bounds so that if N is not a multiple of the block size,
it still works correctly.
*/

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void multiplyByTwoKernel(const float* d_in, float* d_out, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = 2.0f * d_in[idx];
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t N = (size_t)atoll(argv[1]);
    size_t bytes = N * sizeof(float);

    // Host memory allocation
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // Device memory allocation
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Kernel launch
    size_t gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    multiplyByTwoKernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = 2.0f * h_in[i];
        if (fabs(h_out[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n",
                    i, h_out[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Success: All %zu elements multiplied by 2 correctly.\n", N);
    }

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```