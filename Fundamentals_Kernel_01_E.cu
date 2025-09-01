/*
Aim of the program (exact wording from the prompt):
Add `cudaDeviceSynchronize()` after the launch and print 'Kernel synchronized!' after it.

Thinking and reasoning about the implementation:
- The prompt asks to modify a CUDA program by adding a synchronization call after a kernel launch and printing a message afterwards.
- Since no original code is provided, I will write a minimal complete CUDA program that demonstrates this behaviour.
- The program will:
  1. Allocate a small array on the host.
  2. Copy it to device memory.
  3. Launch a simple kernel that increments each element.
  4. Immediately after the launch, call `cudaDeviceSynchronize()` to wait for the kernel to finish.
  5. Print "Kernel synchronized!" to indicate the sync was successful.
  6. Copy the results back and print them for verification.
- All error checks are minimal but sufficient for a small demo.
- The code is self‑contained and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_one(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    int h_arr[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    int *d_arr = NULL;
    size_t size = N * sizeof(int);
    cudaError_t err;

    err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Launch kernel: one block with N threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Add the requested synchronization and print statement
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }
    printf("Kernel synchronized!\n");

    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    printf("Resulting array: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
