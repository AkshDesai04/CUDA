```cpp
/*
Aim: Write a naive matrix transpose kernel where thread (x, y) reads input[y][x] and writes to output[x][y]. Assuming row-major layout, analyze why the reads from input will be uncoalesced (strided) and hurt performance.

Thinking:
To illustrate the uncoalesced read pattern, we will write a simple CUDA kernel where each thread with indices (tx, ty) reads the element at input[ty][tx] (i.e. row ty, column tx) and writes it to output[tx][ty] (i.e. row tx, column ty). In a row-major layout, consecutive elements in the same row are stored contiguously. However, when each thread in a warp reads input[ty][tx] with a fixed tx but varying ty (i.e., each thread reads elements from different rows but the same column), the memory accesses are strided by the matrix width. This means that threads in the same warp will access memory addresses that are not contiguous, causing the GPU to issue multiple memory transactions per warp and leading to low memory throughput. In contrast, the write pattern to output is coalesced because threads in the same warp write to contiguous memory locations (same row of the transposed matrix). This asymmetry (coalesced writes but uncoalesced reads) demonstrates how naive transpose kernels can suffer from poor performance. The code below allocates a simple matrix, runs the naive transpose kernel, and measures the elapsed time. */

#include <cstdio>
#include <cuda_runtime.h>

// Matrix dimensions
const int M = 1024;  // number of rows in input (and columns in output)
const int N = 1024;  // number of columns in input (and rows in output)

// Naive transpose kernel
__global__ void transpose_naive(const float* __restrict__ input, float* __restrict__ output, int width, int height)
{
    // Compute global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index in input
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index in input

    if (x < width && y < height) {
        // Read from input[row=y, col=x] and write to output[row=x, col=y]
        output[x * height + y] = input[y * width + x];
    }
}

int main()
{
    // Size in bytes
    size_t size = M * N * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize input matrix with some values
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and time it
    cudaEventRecord(start);
    transpose_naive<<<gridDim, blockDim>>>(d_input, d_output, N, M);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive transpose kernel elapsed time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Optional: verify a few elements
    bool correct = true;
    for (int i = 0; i < 10 && correct; ++i) {
        int row = i / N;
        int col = i % N;
        float val = h_output[col * M + row];
        if (val != h_input[row * N + col]) {
            printf("Mismatch at (%d,%d): expected %f, got %f\n",
                   row, col, h_input[row * N + col], val);
            correct = false;
        }
    }
    if (correct) printf("Transpose verification passed for first 10 elements.\n");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```