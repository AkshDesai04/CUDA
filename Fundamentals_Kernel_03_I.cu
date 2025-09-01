/*
Aim: Launch with one block of N threads. Pass an integer array `int* out` to the kernel. Have each thread write its `threadIdx.x` to `out[threadIdx.x]`.

My thinking: 
- The kernel will simply write its thread index into the provided array. 
- We allocate a device array of size N, launch the kernel with one block and N threads, then copy back the data to host for verification. 
- We use simple error checking after memory operations and kernel launch. 
- The program is self‑contained and can be compiled with `nvcc program.cu -o program`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes threadIdx.x into out[threadIdx.x]
__global__ void writeThreadIdx(int *out)
{
    int idx = threadIdx.x;
    out[idx] = idx;
}

int main(void)
{
    const int N = 256;                 // Number of threads (also array size)
    int *h_out = NULL;                 // Host array
    int *d_out = NULL;                 // Device array

    // Allocate host memory
    h_out = (int *)malloc(N * sizeof(int));
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch kernel: 1 block, N threads
    writeThreadIdx<<<1, N>>>(d_out);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", err);
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (device to host) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Print the results
    printf("Thread indices written by kernel:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // Clean up
    cudaFree(d_out);
    free(h_out);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
