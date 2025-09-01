```cuda
/*
Aim of the program:
For the GPU timing, use `cudaEvent`s around the kernel launch only.

Thinking:
The task requires a CUDA program that demonstrates how to time a kernel launch
using cudaEvent.  I chose a classic example of vector addition because it is
simple, yet it exercises device memory allocation, memory copy, kernel
launch, and timing.  The program will:

1. Allocate host arrays for two input vectors and one output vector.
2. Allocate corresponding device arrays.
3. Copy input data from host to device.
4. Create two cudaEvent objects: one to mark the start of the kernel launch
   and one to mark its end.  These events are recorded immediately before
   and after the kernel launch, as requested.
5. Launch a simple vector addition kernel.
6. Record the end event, synchronize, and compute the elapsed time in
   milliseconds.
7. Copy the result back to the host.
8. Verify a few elements to ensure correctness.
9. Clean up all allocated memory and destroy the events.

All error checks are performed via a helper macro that aborts on failure.
This keeps the code concise and ensures that any CUDA errors are caught
immediately.  The kernel uses a 1D grid of 1D blocks, each block handling
multiple elements based on the total size and block dimension.

The code is fully self‑contained and can be compiled with:
    nvcc -o vector_add vector_add.cu
and executed to see the timing output.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Define kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time in milliseconds
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification failed at index %d: expected %f, got %f\n",
                    i, expected, h_c[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Result verification passed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```