```c
/*
Aim of the program:
Implement a parallel reduction (e.g., sum) within a single block using shared memory. Each thread loads an element into shared memory. Then, in a loop, threads add elements together, halving the number of active threads in each step with __syncthreads() in between. The final sum is in shared_mem[0].

Thinking:
To perform a parallel reduction efficiently in CUDA, we use shared memory to store intermediate results and synchronize threads after each reduction step. The kernel will be launched with a single block, and the number of threads per block will be a power of two for simplicity. Each thread will load its corresponding element from the global input array into shared memory; if the input size is smaller than the block size, threads with indices beyond the input size will load zero. Then, we iteratively halve the number of active threads: at each step, a thread with index < stride will add the element at index + stride to its own element. After each addition we call __syncthreads() to ensure all writes are visible. After the loop finishes, the total sum of the block resides in shared_mem[0], which is written to a global output array.

The host code allocates a sample array, copies it to the device, launches the kernel, and copies back the result. For demonstration we use a simple vector of 256 elements and a block of 256 threads. We include basic error checking. The program prints the computed sum and compares it with the CPU-computed reference sum for verification. This demonstrates the reduction works correctly.

The code is self-contained in a single .cu file, ready to compile with nvcc. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256   // Must be a power of two

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel performing parallel reduction sum within a single block
__global__ void blockReduceSum(const float *input, float *output, int n)
{
    // Shared memory allocation
    __shared__ float shared_mem[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    // Load data into shared memory (handle case when n < blockDim)
    shared_mem[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // The result is in shared_mem[0]
    if (tid == 0) {
        output[0] = shared_mem[0];
    }
}

int main(void)
{
    const int n = 256; // Number of elements to sum
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(bytes);
    float h_output = 0.0f;
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1.0f; // Simple test: sum should be n
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Launch kernel with one block
    blockReduceSum<<<1, THREADS_PER_BLOCK>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference sum on CPU
    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_sum += h_input[i];
    }

    // Print results
    printf("GPU sum:   %f\n", h_output);
    printf("CPU sum:   %f\n", cpu_sum);
    printf("Difference: %f\n", fabsf(h_output - cpu_sum));

    // Clean up
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
```