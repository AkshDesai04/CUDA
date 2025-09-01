/*
Add error checking to the `cudaMalloc` and `cudaMemcpy` calls.

[Thinking]
1. Create a simple CUDA program that demonstrates allocation, memory copy, kernel launch, and cleanup.
2. Wrap all CUDA runtime calls (cudaMalloc, cudaMemcpy, cudaFree, cudaLaunchKernel, cudaDeviceSynchronize) with an error checking macro.
3. Implement a helper function that prints the CUDA error string and terminates the program if an error occurs.
4. Use a small kernel that performs element‑wise addition of two arrays.
5. Allocate host memory, initialize it, allocate device memory, copy data to the device, launch the kernel, copy results back, and free all resources.
6. Ensure that every cudaMalloc and cudaMemcpy is guarded by the error checking macro.
*/

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Simple element-wise addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate and initialize host arrays
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory with error checking
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy host data to device with error checking
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel completion

    // Copy result back to host with error checking
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: %f != %f\n", i, h_c[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("First 10 results verified successfully.\n");
    } else {
        printf("Found %d errors.\n", errors);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
