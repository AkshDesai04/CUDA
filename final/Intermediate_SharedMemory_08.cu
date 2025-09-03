/*
Implement a parallel reduction (e.g., sum) within a single block using shared memory. Each thread loads an element into shared memory. Then, in a loop, threads add elements together, halving the number of active threads in each step with __syncthreads() in between. The final sum is in shared_mem[0].

Thinking:
- The goal is to write a CUDA kernel that performs a reduction of an array of integers using shared memory within one block.
- Each thread will load one element of the input array into a shared memory array `sdata`. If the number of elements is less than the block size, threads beyond the array size will load zero.
- After loading, we use a loop that starts with the block's thread count divided by two. In each iteration, only the first `s` threads (where `s` starts at blockDim.x/2 and halves each iteration) will add the value of the element `s` positions ahead to their own position in `sdata`.
- We must synchronize with `__syncthreads()` after each addition step to ensure all reads are complete before the next write.
- After the loop, thread 0 will write the result `sdata[0]` to the output array.
- For simplicity, we will work with integers, use a single block, and ensure that the block size is a power of two.
- The host code will:
  * Allocate input and output arrays on the host.
  * Allocate device memory and copy the input to device.
  * Launch the kernel with one block and appropriate shared memory size.
  * Copy back the result and optionally verify against a CPU-computed sum.
  * Free all allocated memory.
- Error checking will be added after CUDA calls to catch and report any issues.

This implementation demonstrates the typical pattern for intra-block reduction using shared memory in CUDA, following the specified steps and synchronization requirements.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for checking CUDA errors
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel for parallel reduction (sum) within a single block
__global__ void reduceSumKernel(const int *input, int *output, int n) {
    // Allocate shared memory dynamically
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;

    // Load input elements into shared memory
    if (tid < n) {
        sdata[tid] = input[tid];
    } else {
        // If the number of elements is less than block size, pad with zeros
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    // Each iteration halves the number of active threads
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // Ensure all additions are done before next stride
    }

    // The result is now in sdata[0]
    if (tid == 0) {
        output[0] = sdata[0];
    }
}

int main(void) {
    const int N = 1024;  // Number of elements in the input array
    const int BLOCK_SIZE = 512;  // Must be a power of two and >= N

    // Host allocation
    int *h_input = (int *)malloc(N * sizeof(int));
    int h_output = 0;
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host input array\n");
        return EXIT_FAILURE;
    }

    // Initialize host input with some values, e.g., 0..N-1
    for (int i = 0; i < N; ++i) {
        h_input[i] = i + 1;  // Sum will be N*(N+1)/2
    }

    // Device allocation
    int *d_input = NULL;
    int *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_output, sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: 1 block, BLOCK_SIZE threads, shared memory of BLOCK_SIZE * sizeof(int)
    reduceSumKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure kernel completion
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Compute CPU sum for verification
    long long cpu_sum = 0;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_input[i];
    }

    printf("GPU reduced sum: %d\n", h_output);
    printf("CPU sum       : %lld\n", cpu_sum);

    // Clean up
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
