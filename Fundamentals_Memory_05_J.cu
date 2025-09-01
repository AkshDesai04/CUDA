/*
Aim: Forget to call `cudaFree`. Run the program. This is a memory leak.

Thinking:
The purpose of this program is to demonstrate how a CUDA application can leak GPU memory
by allocating device memory with `cudaMalloc` and never freeing it with `cudaFree`. 
The code performs a simple initialization of an array on the GPU using a kernel,
copies the data back to host memory, prints a few elements, and then exits.
The device pointer is intentionally not freed; likewise the device is not reset with
`cudaDeviceReset`. This omission results in the allocated GPU memory remaining reserved
until the process terminates, which is a classic memory leak scenario in CUDA
programs. The code is intentionally minimal to highlight the missing `cudaFree`
and to keep the example clear and focused on the leak itself.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initArray(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = idx;
    }
}

int main(void) {
    const int n = 1 << 20;            // 1 million elements
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel to initialize the array
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy back to host
    int *h_arr = (int *)malloc(n * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    err = cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print first 10 elements
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up host memory
    free(h_arr);

    // Intentional leak: no cudaFree(d_arr)
    // Also no cudaDeviceReset to release device resources

    return 0;
}
