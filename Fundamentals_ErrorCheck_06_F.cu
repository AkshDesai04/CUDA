```cpp
/* 
In a program with multiple, sequential kernel launches, place the check after each one to pinpoint which one is failing. 

Thinking:
- We want a simple CUDA C program that launches several kernels in sequence.
- After each kernel launch, we need to check for errors using cudaGetLastError() and cudaDeviceSynchronize().
- We'll create three simple kernels: one that adds 1 to each element, one that multiplies by 2, and one that subtracts 3.
- After each launch, we check for errors; if an error occurs, we print a message indicating which kernel failed.
- Additionally, we can introduce an intentional error (e.g., out-of-bounds write) in the fourth kernel to demonstrate error detection.
- We wrap the error checking logic in a helper macro for clarity.
- The program allocates device memory, launches kernels, copies back results, and cleans up.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

#define CHECK_CUDA_KERNEL(kernel, ...)                                     \
    do {                                                                    \
        kernel;                                                            \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "Kernel error after %s (%s:%d): %s\n",          \
                    #kernel, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
        err = cudaDeviceSynchronize();                                      \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "Synchronize error after %s (%s:%d): %s\n",     \
                    #kernel, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

__global__ void addOne(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] += 1;
}

__global__ void multiplyByTwo(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] *= 2;
}

__global__ void subtractThree(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] -= 3;
}

/* Intentionally erroneous kernel: writes out of bounds */
__global__ void errorKernel(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* Deliberate out-of-bounds write when idx == n */
    d_data[idx] = 42;
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(int);
    int *h_data = (int *)malloc(size);
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));
    CHECK_CUDA(cudaMemset(d_data, 0, size));

    // Launch addOne kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    CHECK_CUDA_KERNEL(addOne<<<gridSize, blockSize>>>(d_data, N), "addOne");

    // Launch multiplyByTwo kernel
    CHECK_CUDA_KERNEL(multiplyByTwo<<<gridSize, blockSize>>>(d_data, N), "multiplyByTwo");

    // Launch subtractThree kernel
    CHECK_CUDA_KERNEL(subtractThree<<<gridSize, blockSize>>>(d_data, N), "subtractThree");

    // Launch intentionally erroneous kernel to trigger an error
    CHECK_CUDA_KERNEL(errorKernel<<<gridSize, blockSize>>>(d_data, N), "errorKernel");

    // Copy back results (should not reach here due to error)
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Print first 10 results
    for (int i = 0; i < 10; ++i)
        printf("h_data[%d] = %d\n", i, h_data[i]);

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    return 0;
}
```