```cuda
/* 
Aim: Implement a parallel reduction (e.g., sum) within a single block using shared memory. 
Each thread loads an element into shared memory. Then, in a loop, threads add elements together, 
halving the number of active threads in each step with __syncthreads() in between. 
The final sum is in shared_mem[0].

Thinking:
1. The reduction algorithm works best when the number of threads is a power of two, 
   because we can halve the active threads each iteration.
2. Each thread will copy one element from global memory into a shared memory array 
   (sdata). If the input array is smaller than the number of threads, we pad with zeros. 
3. After the initial load, we perform the classic parallel reduction: 
   - For s = blockDim.x/2 down to 1, each thread with index < s adds sdata[tid] and sdata[tid+s].
   - __syncthreads() is used after each addition step to ensure all writes are visible before 
     the next step.
4. After the loop, thread 0 contains the block’s sum in sdata[0]. This value is written 
   to global memory as the output.
5. In the host code, we allocate an input array, fill it with test data, copy it to the 
   device, launch the kernel with a single block of size equal to the number of elements, 
   copy back the result, and verify it against a CPU-computed sum.
6. The kernel uses dynamic shared memory (extern __shared__ float sdata[]) and we specify 
   the size at launch as blockDim.x * sizeof(float).
7. Basic error checking macros are provided for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel: parallel reduction (sum) within a single block
__global__ void reduceSum(const float *input, float *output, int N)
{
    // Allocate shared memory dynamically
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    // Load data into shared memory; pad with zero if out of bounds
    if (tid < N)
        sdata[tid] = input[tid];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Parallel reduction: halve the active thread count each iteration
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); // Ensure all additions for this stride are done
    }

    // The result is in sdata[0]
    if (tid == 0)
        *output = sdata[0];
}

int main(void)
{
    const int N = 1 << 20;  // Number of elements (must be <= blockDim.x for single block)
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with some values (e.g., 1.0f)
    for (int i = 0; i < N; ++i)
        h_input[i] = 1.0f;  // Simple case: sum should be N

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel: one block with N threads, dynamic shared memory
    dim3 blockDim(N);
    dim3 gridDim(1);
    size_t sharedMemSize = N * sizeof(float);
    reduceSum<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i)
        cpu_sum += h_input[i];

    printf("GPU sum:   %.2f\n", h_result);
    printf("CPU sum:   %.2f\n", cpu_sum);
    printf("Difference: %.2e\n", fabs(h_result - cpu_sum));

    // Clean up
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```