```cpp
/*
Aim: Modify the program to allocate and free the memory in a loop 100 times.

Thinking:
The task is to take an existing CUDA program that allocates device memory, does a simple operation (like vector addition), and frees that memory, and modify it so that the allocation and deallocation happen inside a loop that runs 100 times. Each iteration should independently allocate the device memory, perform the operation, and then free the memory. This will help to test the robustness of memory management and can also serve as a benchmark for allocation overhead.

Key points to consider:
1. The loop counter is 100, so we need to perform the entire allocation–execution–free cycle 100 times.
2. We must ensure that we do not leak memory: every `cudaMalloc` must be matched with a `cudaFree` in the same iteration.
3. To keep the code simple, we will allocate a small fixed-size array (e.g., 1024 elements) for each iteration.
4. We'll use a trivial CUDA kernel that adds two vectors element‑wise.
5. Error checking after each CUDA API call is important; if an error occurs we’ll print it and exit.
6. We’ll use `cudaDeviceSynchronize` after the kernel launch to make sure the operation finishes before freeing memory.
7. The program will compile as a standard CUDA `.cu` file and can be built with `nvcc`.

This code demonstrates how to repeatedly allocate and free device memory in a loop, ensuring correctness and proper cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that adds two vectors
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Simple error checking macro
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void) {
    const int N = 1024;                     // Number of elements
    const size_t size = N * sizeof(float);  // Size in bytes

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Loop 100 times: allocate, compute, free
    for (int iter = 0; iter < 100; ++iter) {
        float *d_a = NULL;
        float *d_b = NULL;
        float *d_c = NULL;

        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void **)&d_a, size));
        CUDA_CHECK(cudaMalloc((void **)&d_b, size));
        CUDA_CHECK(cudaMalloc((void **)&d_c, size));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());          // Check launch errors
        CUDA_CHECK(cudaDeviceSynchronize());     // Wait for kernel to finish

        // Copy result back to host (optional, we just demonstrate)
        CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

        // Verify result (optional)
        for (int i = 0; i < N; ++i) {
            float expected = h_a[i] + h_b[i];
            if (fabs(h_c[i] - expected) > 1e-5f) {
                fprintf(stderr, "Result mismatch at index %d: got %f, expected %f\n",
                        i, h_c[i], expected);
                exit(EXIT_FAILURE);
            }
        }

        // Free device memory
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));

        // Optional: print progress
        if ((iter + 1) % 10 == 0) {
            printf("Iteration %d completed.\n", iter + 1);
        }
    }

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    printf("All 100 iterations completed successfully.\n");
    return 0;
}
```