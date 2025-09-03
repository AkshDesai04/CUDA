```cpp
/* 
Add `cudaDeviceSynchronize()` after the kernel launch and explain in a comment why this is important for timing and error checking.
 
The idea behind this program is to illustrate the proper placement of `cudaDeviceSynchronize()` in a CUDA application. 
The program performs a simple element‑wise addition of two vectors on the GPU. After the kernel is launched, 
`cudaDeviceSynchronize()` is called to block the host until all device work has completed. This ensures that 
any timing measurements taken after the call reflect the actual GPU execution time, not just the time to issue 
the launch command. It also forces any asynchronous CUDA errors (such as invalid memory accesses or launch
configuration errors) to be reported back to the host at this point, allowing reliable error checking.

Below the call, a comment explains these two points in detail: why the synchronization is crucial for accurate
timing and for proper error handling.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// Simple vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // --------------------------------------------------------------------------------
    // Wait for the kernel to finish before proceeding.
    // This is important for two reasons:
    // 1. Timing: `cudaDeviceSynchronize()` blocks the host until all previously issued
    //    device work has completed. Without it, a subsequent CPU timer would measure
    //    only the time taken to launch the kernel, not the actual execution time.
    // 2. Error checking: many CUDA errors are reported asynchronously. If we
    //    immediately query the error status or perform a memory copy, we may
    //    miss errors that occur during kernel execution. Synchronizing forces
    //    all pending operations to complete and any errors to surface, allowing
    //    reliable error detection with `cudaGetLastError()` or `cudaPeekAtLastError()`.
    // --------------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure kernel has finished and errors are caught

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results (simple check)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            fprintf(stderr, "Mismatch at index %d: %f != %f + %f\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }

    printf("%s\n", success ? "Success!" : "Failure!");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```