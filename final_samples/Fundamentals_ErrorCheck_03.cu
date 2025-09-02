```cpp
/*
Kernel launches do not return an error code directly. Add error checking for a kernel launch by calling `cudaPeekAtLastError()` immediately after the launch and `cudaDeviceSynchronize()` sometime later. Wrap both of these calls with your `CHECK()` macro.

Below is a minimal CUDA program that demonstrates this approach. 
We define a simple kernel that adds two vectors and writes the result to a third vector. 
The program allocates device memory, copies data to the device, launches the kernel, and then checks for errors immediately after the launch using `cudaPeekAtLastError()` and later after the kernel completes using `cudaDeviceSynchronize()`. 
The `CHECK` macro is used to wrap these calls, printing an informative message and terminating the program if an error occurs. 
This showcases the recommended pattern for robust error handling in CUDA kernel launches.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, size));
    CHECK(cudaMalloc((void**)&d_B, size));
    CHECK(cudaMalloc((void**)&d_C, size));

    // Copy host to device
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Immediate error check after kernel launch
    CHECK(cudaPeekAtLastError());

    // Copy result back to host
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Synchronize device to ensure all operations are complete
    CHECK(cudaDeviceSynchronize());

    // Verify results (simple check)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition succeeded!\n");
    }

    // Clean up
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```