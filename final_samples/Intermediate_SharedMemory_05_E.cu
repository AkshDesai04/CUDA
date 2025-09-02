```cpp
/*
Aim of the program:
Compare the performance of the boundary-checked kernel on a perfect-multiple size (e.g. 512x512) vs. the original kernel without checks. Is there overhead?

Thinking:
To determine whether the boundary check in a CUDA kernel introduces measurable overhead when the grid size exactly matches the problem dimensions, I will implement two identical kernels that perform element-wise addition of two matrices. One kernel includes an explicit bounds check (`if (x < N && y < M)`) before accessing memory, while the other omits this check. By launching each kernel on a 512x512 array (which is a perfect multiple of the block size 16x16), I can ensure that no thread will ever execute out-of-bounds, thus any extra work performed by the boundary-checked version should be purely the cost of the `if` condition.

The program will:
1. Allocate and initialize two input arrays on the host.
2. Copy them to the device.
3. Launch both kernels, measuring execution time with CUDA events.
4. Copy the results back and verify correctness.
5. Print the elapsed times and the ratio to highlight any overhead.

Because the problem size is modest (262,144 elements) and the kernel is trivial, the difference may be small, but this benchmark demonstrates how to quantify it.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 512
#define M 512
#define BLOCK_SIZE 16

// Kernel with boundary check
__global__ void add_kernel_check(const float *a, const float *b, float *c, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel without boundary check
__global__ void add_kernel_no_check(const float *a, const float *b, float *c, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    c[idx] = a[idx] + b[idx];
}

// Utility for error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main()
{
    const int size = N * M;
    const int bytes = size * sizeof(float);

    // Host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_check = (float *)malloc(bytes);
    float *h_c_no_check = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < size; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch boundary-checked kernel
    CUDA_CHECK(cudaEventRecord(start));
    add_kernel_check<<<gridDim, blockDim>>>(d_a, d_b, d_c, N, M);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_check = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_check, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c_check, d_c, bytes, cudaMemcpyDeviceToHost));

    // Launch no-boundary-check kernel
    CUDA_CHECK(cudaEventRecord(start));
    add_kernel_no_check<<<gridDim, blockDim>>>(d_a, d_b, d_c, N, M);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_no_check = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_no_check, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c_no_check, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < size; ++i) {
        if (fabs(h_c_check[i] - h_c_no_check[i]) > 1e-5f) {
            errors++;
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: %f vs %f\n", i, h_c_check[i], h_c_no_check[i]);
            }
        }
    }
    if (errors == 0) {
        printf("Results match.\n");
    } else {
        printf("Found %d mismatches.\n", errors);
    }

    printf("Boundary-checked kernel time: %.3f ms\n", ms_check);
    printf("No-check kernel time:          %.3f ms\n", ms_no_check);
    printf("Overhead ratio (check/no-check): %.3f\n", ms_check / ms_no_check);

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c_check);
    free(h_c_no_check);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```