```cuda
/*
 * Aim of the program:
 * Run your leaky program with `cuda-memcheck --leak-check full` and analyze the output.
 *
 * Thinking:
 * The goal is to create a CUDA program that intentionally leaks memory on both the host and the device.
 * The simplest way to leak device memory is to call cudaMalloc and never call cudaFree.
 * For the host side, we can use malloc (or calloc) and never free the pointer.
 * The program will allocate a large array on the device, launch a kernel that performs a trivial operation on the array,
 * and then exit without freeing the device memory. Likewise, we allocate a host array, fill it, copy it to the device,
 * and never free the host buffer. This will cause cuda-memcheck to report a device memory leak and a host memory leak.
 *
 * The kernel itself performs a simple computation (adds 1.0f to each element) to ensure the allocated memory is actually used,
 * preventing the compiler from optimizing away the allocation.
 *
 * The program is intentionally simple: it demonstrates the leak, prints a message, and exits.
 * Running with `cuda-memcheck --leak-check full` will show the leaked device memory and the host memory that was never freed.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_one(float *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1.0f;
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host allocation that will not be freed
    float *h_arr = (float *)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = 0.0f;
    }

    // Device allocation that will not be freed
    float *d_arr;
    cudaError_t err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_arr); // free host memory to avoid crash
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr); // free device memory before exit
        free(h_arr);
        return 1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy back from device: %s\n", cudaGetErrorString(err));
    }

    // Print first 5 elements to verify kernel ran
    printf("First 5 elements after kernel: ");
    for (int i = 0; i < 5; ++i) {
        printf("%f ", h_arr[i]);
    }
    printf("\n");

    // Intentionally do not free h_arr or d_arr to create leaks

    // Reset device to clean up state
    cudaDeviceReset();

    return 0;
}
```