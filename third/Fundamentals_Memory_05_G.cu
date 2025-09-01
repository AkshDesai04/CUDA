```cuda
/*
Aim: What happens if you allocate memory but your program crashes before `cudaFree` is called? Does the memory get reclaimed?

Thinking:
- Device memory is allocated by the CUDA driver on behalf of the process.
- If the process terminates unexpectedly (crash, abort, or explicit exit), the driver
  detects the termination and automatically frees all resources that were allocated
  for that process, including device memory.
- Therefore, the memory is reclaimed when the process ends, even if `cudaFree` was never
  called.
- If the process continues to run but never frees the memory, it remains allocated
  until the process exits or until `cudaFree` is called.
- In practice, you should free memory explicitly, but the driver’s cleanup on process
  exit prevents leaks that would otherwise affect other processes.
- This program demonstrates allocating memory, printing device free memory before
  and after allocation, then intentionally aborting before calling `cudaFree`.  While
  the reclamation occurs when the process exits, we cannot directly observe it from
  within the same process; however, the driver documentation guarantees the
  cleanup.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx * 2;
    }
}

int main() {
    size_t freeMem, totalMem;
    cudaError_t err;

    // Query free memory before allocation
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error querying memory info before allocation: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("Before allocation: Free memory = %zu bytes, Total memory = %zu bytes\n",
           freeMem, totalMem);

    const int N = 1 << 20; // 1M integers
    int *d_arr = NULL;
    err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Query free memory after allocation
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error querying memory info after allocation: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr); // Clean up before exiting on error
        return EXIT_FAILURE;
    }
    printf("After allocation: Free memory = %zu bytes, Total memory = %zu bytes\n",
           freeMem, totalMem);

    // Launch a simple kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();

    // Intentionally abort before calling cudaFree to simulate a crash
    printf("Simulating crash: aborting before cudaFree\n");
    abort(); // This will terminate the process abruptly

    // The following code will not be executed
    // cudaFree(d_arr);
    // cudaDeviceReset();

    return EXIT_SUCCESS;
}
```