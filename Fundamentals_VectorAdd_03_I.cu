```cuda
/*
Aim of the program:
Move the global index calculation and boundary check into a `__device__` helper function.

Thinking:
- The goal is to separate the logic for computing a linear global index from the kernel code and also include a boundary check.
- A helper function `getIdx` will take thread indices (x, y), grid width and height, and compute the linear index.
- It will return -1 if the thread is out of bounds, allowing the kernel to skip invalid work.
- The kernel will then simply call this helper, check for a valid index, and perform its computation.
- We demonstrate this with a simple vector addition kernel operating on 2D data flattened into 1D arrays.
- The main function allocates memory, initializes data, launches the kernel, copies back results, and verifies correctness.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Device helper function to compute linear index and perform boundary check
__device__ int getIdx(int x, int y, int width, int height) {
    if (x >= width || y >= height || x < 0 || y < 0) {
        return -1; // Invalid index
    }
    return y * width + x;
}

// Simple kernel that adds two arrays using the helper function
__global__ void addKernel(const float* a, const float* b, float* c, int width, int height) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // X coordinate
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // Y coordinate

    int idx = getIdx(tx, ty, width, height);
    if (idx >= 0) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 1024;
    const int height = 768;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < size; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(size - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc((void**)&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_c, bytes), "cudaMalloc d_c");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_a to d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_b to d_b");

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    addKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_c to h_c");

    // Verify results
    int errors = 0;
    for (int i = 0; i < size; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return EXIT_SUCCESS;
}
```