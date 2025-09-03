/*
Place a `CHECK(cudaGetLastError())` before your kernel launch to clear any prior unrelated errors.

The idea behind this program is to demonstrate how to clear the CUDA error
queue before launching a kernel.  CUDA operations accumulate errors
internally, and if a prior unrelated error has occurred, it may mask the
error that actually originates from the kernel launch.  By calling
`cudaGetLastError()` immediately before launching the kernel we consume
and discard any pending errors, ensuring that the error status reflects
only the subsequent kernel execution.

The program performs a simple element‑wise addition of two floating‑point
arrays on the GPU.  A macro `CHECK` is defined to wrap CUDA calls and
report errors immediately.  Right before launching the kernel, the
program explicitly calls `CHECK(cudaGetLastError());` to clear the error
queue.  After the kernel launch, we use `cudaDeviceSynchronize()` and
`CHECK` again to catch any errors that arose during the kernel execution.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",       \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple kernel to add two arrays
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));
    CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Clear any previous CUDA errors before launching the kernel
    CHECK(cudaGetLastError()); // This clears the error queue

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK(cudaGetLastError());

    // Synchronize and check for runtime errors
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Result verified successfully.\n");
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}
