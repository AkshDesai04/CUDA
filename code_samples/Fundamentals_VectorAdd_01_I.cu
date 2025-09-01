```cpp
/*
Refactor the host code into functions: `init_vectors()`, `run_kernel()`, `verify_results()`

Thinking:
- I need to write a simple CUDA program that demonstrates refactoring host logic into separate functions.
- The program will allocate two input vectors and one output vector, initialize them on the host, copy them to the device, run a kernel (element-wise addition), copy result back, and verify correctness.
- The host logic will be split into three functions:
  * init_vectors(): allocate host memory, initialize data, allocate device memory, copy data to device.
  * run_kernel(): launch the kernel and optionally measure execution time.
  * verify_results(): copy output from device to host and compare with expected values.
- I will use simple int arrays for clarity, but could use floats too.
- I'll include an error-checking macro to catch CUDA errors.
- The main() function will call these functions in order, and finally free all memory.
- I will provide the full .cu code with necessary includes and definitions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)  // Number of elements
#define BLOCK_SIZE 256

// Error checking macro
#define CUDA_CHECK(err)                                            \
    do {                                                           \
        cudaError_t err__ = (err);                                 \
        if (err__ != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err__));\
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel: element-wise addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Global pointers for host and device data
int *h_A = NULL, *h_B = NULL, *h_C = NULL;
int *d_A = NULL, *d_B = NULL, *d_C = NULL;

// Function prototypes
void init_vectors();
void run_kernel();
void verify_results();
void cleanup();

int main(void)
{
    init_vectors();
    run_kernel();
    verify_results();
    cleanup();
    return 0;
}

// Allocate and initialize host and device vectors
void init_vectors()
{
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
}

// Launch the kernel
void run_kernel()
{
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Optional: Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Verify the results
void verify_results()
{
    size_t size = N * sizeof(int);
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify each element
    for (int i = 0; i < N; ++i) {
        int expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            fprintf(stderr, "Verification failed at index %d: GPU=%d, CPU=%d\n",
                    i, h_C[i], expected);
            exit(EXIT_FAILURE);
        }
    }
    printf("Verification successful: all results match.\n");
}

// Free all allocated memory
void cleanup()
{
    if (h_A) free(h_A);
    if (h_B) free(h_B);
    if (h_C) free(h_C);
    if (d_A) CUDA_CHECK(cudaFree(d_A));
    if (d_B) CUDA_CHECK(cudaFree(d_B));
    if (d_C) CUDA_CHECK(cudaFree(d_C));
}
```