```cpp
/*
Aim of the program:
Place a `CHECK(cudaGetLastError())` before your kernel launch to clear any prior unrelated errors.

Thinking and design:
1. Define a macro `CHECK` that wraps CUDA runtime calls and prints a detailed error message if
   the call fails. This macro will also exit the program on error to simplify debugging.
2. Write a simple kernel (vector addition) that will be launched from `main`.
3. In `main` allocate host and device memory, initialize the host data, and copy it to the device.
4. Intentionally introduce a possible CUDA error prior to the kernel launch. In this example,
   we simply call `cudaGetLastError()` after a previous call that may fail (e.g., `cudaMalloc`
   or `cudaMemcpy`). The returned error is ignored because we want to demonstrate how to clear
   it before launching the kernel.
5. Immediately before the kernel launch, call `CHECK(cudaGetLastError())`. This clears any
   lingering error state so that the kernel launch will not be aborted because of an unrelated
   earlier error.
6. Launch the kernel and perform a `CHECK(cudaGetLastError())` afterwards to verify that the
   kernel executed without errors.
7. Copy the results back to the host and verify correctness.
8. Clean up all allocated memory.

The program demonstrates best practices for error handling in CUDA and illustrates the
use of `cudaGetLastError()` to reset the error state before a kernel launch.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                            \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", \
                    #call, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Host allocations
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));
    CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Intentionally introduce a CUDA error before kernel launch
    // For example, a bad memory copy with wrong size
    // This call will produce an error but we will clear it later
    // Note: In this example we use a dummy size to force an error
    // Uncomment the following line to see the error being cleared
    // CHECK(cudaMemcpy(d_c, d_a, size + 10, cudaMemcpyDeviceToDevice));

    // Clear any prior CUDA errors before launching the kernel
    CHECK(cudaGetLastError()); // This resets the error state

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError()); // Check for errors from kernel launch

    // Copy result back to host
    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification failed at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful. All results verified.\n");
    } else {
        printf("Vector addition failed verification.\n");
    }

    // Clean up
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```