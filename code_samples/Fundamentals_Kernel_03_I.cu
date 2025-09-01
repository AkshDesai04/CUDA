/*
Aim of the program: 
Launch with one block of N threads. Pass an integer array `int* out` to the kernel. Have each thread write its `threadIdx.x` to `out[threadIdx.x]`.

Thinking: 
To fulfill the goal, we create a simple CUDA kernel that writes each thread's index into the output array. We allocate memory on the host, then allocate device memory, and launch the kernel with one block of N threads (`<<<1, N>>>`). After execution, we copy the result back to the host and print it to verify correctness. Error checks are included for kernel launch and memory operations. This program demonstrates basic CUDA memory handling and kernel launch syntax.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void write_thread_indices(int *out)
{
    int idx = threadIdx.x;
    out[idx] = idx;
}

int main(void)
{
    const int N = 256;                     // Number of threads (size of the array)
    int *h_out = (int*)malloc(N * sizeof(int));   // Host array
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    int *d_out = NULL;
    cudaError_t err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_out);
        return 1;
    }

    // Launch kernel with one block of N threads
    write_thread_indices<<<1, N>>>(d_out);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy from device to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    // Print the output array
    printf("Thread indices written to array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return 0;
}
