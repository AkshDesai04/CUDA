```c
/* 
Aim of the program:
Write a kernel that performs the operation in-place. It takes a single pointer `a` and computes `a[i] = a[i] + 10;`.

My thinking:
To satisfy the prompt I will write a minimal CUDA program that defines a kernel performing the in-place addition of 10 to each element of an array. I will allocate an array on the host, copy it to the device, launch the kernel with enough blocks/threads to cover the array, copy the results back, and verify that each element has been incremented by 10. The kernel will include a bounds check to avoid out-of-bounds memory accesses. I will also include basic error handling for CUDA API calls. The program will be self‑contained and can be compiled with nvcc. 
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void addTenKernel(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += 10;
    }
}

int checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int *)malloc(size);
    if (!h_a) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) h_a[i] = i;

    // Allocate device memory
    int *d_a = NULL;
    cudaMalloc((void **)&d_a, size);
    checkCUDAError("cudaMalloc");

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy H2D");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addTenKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);
    checkCUDAError("kernel launch");

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    checkCUDAError("cudaDeviceSynchronize");

    // Copy results back to host
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy D2H");

    // Verify results
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_a[i] != i + 10) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i + 10, h_a[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("All elements incremented correctly by 10.\n");

    // Cleanup
    free(h_a);
    cudaFree(d_a);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```