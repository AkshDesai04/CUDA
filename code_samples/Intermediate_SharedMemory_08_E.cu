/*
At the end of the kernel, only thread 0 should write the final result from `shared_mem[0]` to the global memory output.

This program demonstrates a CUDA kernel that performs a parallel reduction (sum) using shared memory. The kernel is designed so that, after the reduction has completed, only thread 0 in each block writes the result to global memory. This behavior ensures that we avoid multiple threads writing to the same location, which could lead to race conditions or unnecessary memory traffic.

Key design points:

1. **Shared Memory Allocation**  
   We use `extern __shared__` to allocate an array whose size is specified at launch time. This allows the kernel to use as much shared memory as the block dimension requires, avoiding hard‑coded limits.

2. **Thread Work Distribution**  
   Each thread loads one element of the input array (or zero if the index is out of bounds). This guarantees a full coverage of the data while preventing out‑of‑bounds memory accesses.

3. **Parallel Reduction**  
   The classic tree‑based reduction pattern is used. In each iteration, threads with indices less than `s` add the value from `tid + s`. A `__syncthreads()` call synchronizes the threads before the next loop iteration.

4. **Single Write**  
   After the reduction loop, only the thread with `tid == 0` writes the shared memory result to the global output array. This is the only thread that performs a write to global memory, satisfying the requirement.

5. **Host Code**  
   The host side allocates input and output arrays, initializes the input with random integers, and copies them to the device. It then launches the kernel with a single block for simplicity (since the sum fits within one block). Finally, it copies the result back to the host and prints it.

6. **Error Checking**  
   A helper macro `CUDA_CHECK` is used to check the return status of CUDA runtime calls. This helps catch any issues early.

The code below is self‑contained and can be compiled with `nvcc`:
```
nvcc -arch=sm_61 -o sum_shared sum_shared.cu
```
and then executed. The output will be the sum of all input elements.
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void sumKernel(const int *input, int *output, int n) {
    extern __shared__ int shared_mem[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element into shared memory (zero if out of bounds)
    shared_mem[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Only thread 0 writes the block's partial sum to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

int main() {
    const int N = 1 << 20;          // 1 million elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    int *h_input  = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(blocksPerGrid * sizeof(int));
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random integers
    srand((unsigned)time(nullptr));
    for (int i = 0; i < N; ++i) {
        h_input[i] = rand() % 10; // small numbers for easier sum
    }

    // Allocate device memory
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, blocksPerGrid * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(int);
    sumKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    // Final sum on host
    long long totalSum = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        totalSum += h_output[i];
    }

    printf("Sum of %d elements is: %lld\n", N, totalSum);

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
