```cpp
/*
Aim of the program:
Use pinned host memory (`cudaMallocHost`) for your data buffers.

Thinking:
- The goal is to demonstrate how to allocate host memory that is pinned (page-locked) so that CUDA can perform faster memory transfers between host and device.
- We'll allocate a simple array of floats on the host using cudaMallocHost. This ensures the memory is pinned.
- We'll allocate device memory of the same size with cudaMalloc.
- We'll copy data from the pinned host buffer to the device buffer using cudaMemcpy.
- We'll launch a simple kernel that increments each element by 1.0f.
- We'll copy the result back from the device to the pinned host buffer.
- Finally, we'll verify the result on the host.
- Error checking is performed after each CUDA API call and kernel launch.
- Resources (pinned host memory, device memory) are freed before program exit.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Simple kernel: add 1.0f to each element
__global__ void addOne(float *d_data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] += 1.0f;
    }
}

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate pinned host memory
    float *h_data = NULL;
    CHECK_CUDA(cudaMallocHost((void**)&h_data, size));

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_data, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOne<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure kernel finished

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_data[i] - (static_cast<float>(i) + 1.0f)) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: host %f, expected %f\n",
                    i, h_data[i], static_cast<float>(i) + 1.0f);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Verification passed.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFreeHost(h_data));

    return 0;
}
```