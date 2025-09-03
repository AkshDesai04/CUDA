/*
Implement a vector squaring kernel using a grid-stride loop.

Thinking:
The goal is to create a CUDA program that demonstrates a vector squaring operation, where each element of an input array is squared and stored in an output array. The kernel must use a grid-stride loop to correctly handle cases where the total number of elements exceeds the total number of threads launched. A grid-stride loop is a common pattern that allows each thread to process multiple elements by incrementing its index by the total number of threads in the grid on each iteration.

Design decisions:
1. Use the CUDA runtime API with a simple C interface to keep the code straightforward.
2. Implement a `square` kernel that takes input and output pointers and the length of the vector. Inside the kernel, each thread calculates its global index and then uses a loop:  
   `for (int i = idx; i < N; i += stride)`  
   where `stride` is the total number of threads (`blockDim.x * gridDim.x`).
3. Allocate host memory, initialize the input vector with sequential values for easy verification, and copy it to the device.
4. Choose a block size of 256 threads, which is a typical choice for GPUs. The grid size will be calculated as `(N + blockDim.x - 1) / blockDim.x` to cover all elements; the grid‑stride loop then ensures proper coverage even if the total number of elements is large.
5. Add a simple error checking macro `CUDA_CHECK` to wrap CUDA API calls.
6. After kernel execution, copy the result back to the host, print the first few values for validation, and free all allocated memory.

The final program compiles with `nvcc` and runs on any CUDA-enabled device, demonstrating vector squaring with a grid‑stride loop.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that squares each element of the input vector using a grid‑stride loop
__global__ void square(const float* __restrict__ x,
                       float* __restrict__ y,
                       int N) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // Grid‑stride loop
    for (int i = idx; i < N; i += stride) {
        float val = x[i];
        y[i] = val * val;
    }
}

int main() {
    const int N = 1 << 20;  // 1,048,576 elements (~4 MB for float)
    const int sizeInBytes = N * sizeof(float);

    // Allocate host memory
    float* h_x = (float*)malloc(sizeInBytes);
    float* h_y = (float*)malloc(sizeInBytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with sequential values
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, sizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_y, sizeInBytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeInBytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    square<<<gridSize, blockSize>>>(d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeInBytes, cudaMemcpyDeviceToHost));

    // Verify and print first 10 results
    printf("First 10 results (x[i]^2):\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_y[%d] = %f (expected %f)\n", i, h_y[i], h_x[i] * h_x[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);

    return 0;
}
