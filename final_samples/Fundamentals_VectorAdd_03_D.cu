```cuda
/*
Aim: Remove the `if (index < N)` check and run the program for a non-multiple size with `cuda-memcheck`. Observe the out-of-bounds access errors.

Thinking:
- We create a simple kernel that writes each thread's global index to a device array.
- Allocate an array of size N on the device.
- Launch a grid where the total number of threads (blocks * threadsPerBlock) is not a multiple of N.
- Remove the usual bounds check `if (index < N)` so that threads with indices >= N write out of bounds.
- After kernel launch, we copy back the array and print a few values.
- The program is intended to be run under `cuda-memcheck` to catch the out-of-bounds write errors.
- Error handling for CUDA calls is included for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10                     // Array size (not a multiple of block size)
#define THREADS_PER_BLOCK 4      // Threads per block (4 * 3 = 12 > N)

__global__ void writeIndices(int *d_arr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally no bounds check to induce out-of-bounds access
    d_arr[idx] = idx;
}

int main()
{
    int *d_arr;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with more threads than array elements
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // ceil division
    // Intentionally launch an extra block to create out-of-bounds indices
    blocks += 1; // Now blocks * THREADS_PER_BLOCK > N
    writeIndices<<<blocks, THREADS_PER_BLOCK>>>(d_arr);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy back results to host
    int h_arr[N];
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Print array contents
    printf("Device array contents (first %d elements):\n", N);
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    cudaFree(d_arr);
    return 0;
}
```