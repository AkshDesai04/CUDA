```cuda
/*
Aim of the program: Add the `CHECK` macro around a call to `cudaGetLastError()`

Thinking process:
- The task is to demonstrate the use of a `CHECK` macro that wraps CUDA API calls and checks for errors.
- We will implement a simple CUDA kernel (vector addition) and use the macro after the kernel launch.
- The macro will call the CUDA API function, capture the returned error code, and if it is not `cudaSuccess`, print an error message with file and line information and terminate the program.
- After the kernel launch, we will use `CHECK(cudaGetLastError());` to capture any launch-time errors. This satisfies the requirement to add the `CHECK` macro around that specific call.
- Additionally, we will call `cudaDeviceSynchronize()` and check its result to demonstrate error handling after synchronization.
- The program includes necessary headers, defines the macro, and provides a `main` function that allocates memory, launches the kernel, checks errors, and cleans up.
- The code is fully self-contained and can be compiled with `nvcc`.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Add the CHECK macro around cudaGetLastError()
    CHECK(cudaGetLastError()); // This line satisfies the requirement

    // Wait for GPU to finish
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Simple verification
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            if (errors < 10) {
                printf("Mismatch at %d: %f != %f\n", i, h_c[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful, no errors found.\n");
    } else {
        printf("Vector addition found %d errors.\n", errors);
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```