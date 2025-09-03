```cpp
/*
Aim of the program:
Test your modified kernel with non-multiple sizes like 500x500 and verify correctness.

Thinking:
The goal is to demonstrate that a CUDA kernel correctly handles matrix operations
even when the matrix dimensions are not multiples of the block size.  A typical
pitfall is that kernels assume a fixed grid size that divides evenly into the
problem domain, leading to out-of-bounds accesses or missed elements when the
size is non‑multiples of the block dimension.

To verify correctness, we will:
1. Allocate two 500×500 input matrices on the host and initialize them with
   deterministic values (e.g., A[i][j] = i+j, B[i][j] = i-j).
2. Allocate space for the result matrix C on the host.
3. Allocate corresponding device buffers and copy inputs to the device.
4. Launch a simple element‑wise addition kernel that uses a 16×16 thread block
   and a grid sized by ceil(rows/16) × ceil(cols/16).  The kernel will perform
   bounds checks so that it safely handles the 500×500 case.
5. Copy the result back to the host and compare each element with the expected
   value computed on the CPU.
6. Print a message indicating success or failure.

The kernel is deliberately simple but includes the necessary boundary checks,
showing that it works for arbitrary sizes.  The program also prints timing
information for completeness.

Note that the program uses standard CUDA runtime API calls and assumes a CUDA
compatible compiler (nvcc).  No external libraries are required.  All
matrices are stored in row-major order.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

const int WIDTH  = 500;   // Matrix width
const int HEIGHT = 500;   // Matrix height
const int BLOCK_SIZE = 16; // Threads per block in each dimension

// Kernel: element-wise addition C = A + B
__global__ void add_matrices(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = WIDTH * HEIGHT;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            int idx = i * WIDTH + j;
            h_A[idx] = (float)(i + j);
            h_B[idx] = (float)(i - j);
            h_ref[idx] = h_A[idx] + h_B[idx]; // expected result
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    add_matrices<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH, HEIGHT);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Kernel execution time: %.6f ms\n", elapsed.count() * 1000.0);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_ref[i]) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: device %f, expected %f\n",
                    i, h_C[i], h_ref[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Verification PASSED: Result is correct.\n");
    } else {
        printf("Verification FAILED: Result is incorrect.\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
```