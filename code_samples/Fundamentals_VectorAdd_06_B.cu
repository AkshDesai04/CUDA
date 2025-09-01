```cpp
/*
The original SAXPY is in-place, modifying Y. Implement this version: Y[i] = a * X[i] + Y[i].

Thinking:
To satisfy the requirement, we need a CUDA program that performs the SAXPY operation
in-place on array Y, i.e. each element of Y is updated as Y[i] = a * X[i] + Y[i].
The program will allocate two arrays on the host: X and Y, fill them with
some test data, copy them to device memory, launch a kernel that performs the
computation, copy the results back, and finally print a few elements to
confirm the operation worked.

The kernel is straightforward: each thread computes the index, checks bounds,
and updates Y[idx].  Host code performs all memory allocations, data transfers,
and kernel launch.  Simple error checking is included via a macro that aborts
on CUDA errors.

No external dependencies are required beyond the CUDA runtime.  The code
compiles with `nvcc saxpy.cu -o saxpy` and runs on any GPU that supports
CUDA.

*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// SAXPY kernel: Y[i] = a * X[i] + Y[i]
__global__ void saxpy_kernel(int n, float a, const float *X, float *Y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main(int argc, char *argv[]) {
    // Default problem size and scalar a
    const int N = 1 << 20;          // 1M elements
    const float a = 2.5f;

    // Allocate host memory
    float *h_X = (float*)malloc(N * sizeof(float));
    float *h_Y = (float*)malloc(N * sizeof(float));
    if (!h_X || !h_Y) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i) * 0.001f;   // X[i] = 0.001 * i
        h_Y[i] = static_cast<float>(i) * 0.002f;   // Y[i] = 0.002 * i
    }

    // Allocate device memory
    float *d_X = nullptr;
    float *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, a, d_X, d_Y);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result (print first 10 elements)
    printf("First 10 results of Y after SAXPY (Y = a*X + Y):\n");
    for (int i = 0; i < 10; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    // Clean up
    free(h_X);
    free(h_Y);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    return EXIT_SUCCESS;
}
```