/*
Add error checking to the `cudaMalloc` and `cudaMemcpy` calls.

Thinking:
- In CUDA programming, runtime API functions such as `cudaMalloc` and `cudaMemcpy` return a `cudaError_t` that indicates whether the operation succeeded.  
- If an error occurs and is not checked, the program may continue executing with invalid data, leading to hard‑to‑debug crashes or incorrect results.  
- A common pattern is to wrap these calls in a macro or helper function that checks the return value, prints a descriptive message (using `cudaGetErrorString`), and aborts the program if the call fails.  
- The macro `CHECK_CUDA(call)` will perform this check.  
- The example below demonstrates a simple vector addition kernel and shows the use of `CHECK_CUDA` for all `cudaMalloc` and `cudaMemcpy` operations.  
- Error checking is performed immediately after each call so that any failure is reported with the exact location and reason.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA runtime API calls for errors
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                      \
        }                                                             \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const int N = 1 << 20;          // Number of elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory with error checking
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy input data from host to device with error checking
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host with error checking
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result (optional)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Mismatch at index %d: %f != %f + %f\n", i, h_c[i], h_a[i], h_b[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition succeeded for %d elements.\n", N);
    }

    // Free device memory with error checking
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
