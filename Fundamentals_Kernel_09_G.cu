/*
Aim of the program:
Create a __device__ function that is recursive, like a factorial function. Try calling it from a kernel. Be aware of stack limitations.

Thinking:
The requirement is to write a CUDA program (.cu file) that demonstrates a recursive device function (factorial) and calls it from a kernel. 
CUDA allows recursive device functions but they consume stack space per thread, and the default device stack size is usually very small (around 4KB). 
To safely use recursion we need to set a larger stack limit with cudaDeviceSetLimit(cudaLimitStackSize, <size>). 
We’ll keep the recursion depth small (e.g. computing factorial for numbers up to 10) so the stack usage remains modest, but we still set the limit explicitly to illustrate the technique.

Implementation plan:
1. Define a __device__ unsigned long long factorial(unsigned int n) that returns 1 for n <= 1 and otherwise n * factorial(n-1). This function is straightforward recursion.
2. Write a kernel that accepts an array of integers 'nums' and an output array 'results'. For each thread, compute factorial(nums[threadIdx.x]) and store in results[threadIdx.x]. Also print the result with device printf for demonstration.
3. In the host main function:
   - Allocate host arrays for input numbers (e.g. {5, 7, 9}) and result array.
   - Allocate device memory for these arrays.
   - Copy input numbers to device.
   - Set a larger device stack limit using cudaDeviceSetLimit.
   - Launch the kernel with enough blocks/threads to cover all input numbers.
   - Copy results back to host.
   - Print the results from host.
   - Free device memory.
4. Use error checking macros to make the code robust.

Edge cases:
- If the factorial argument is large, the recursion depth may exceed stack limit; we keep small numbers.
- The kernel uses device printf, which requires the device to support printf (arch >= 2.0). We target a modern GPU (e.g. sm_50+).

The final .cu file contains all of the above with necessary includes and CUDA error checking. The program is self‑contained and can be compiled with nvcc.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for CUDA error checking */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Device recursive factorial function */
__device__ unsigned long long factorial(unsigned int n) {
    if (n <= 1) return 1ULL;
    /* Recursive call */
    return n * factorial(n - 1);
}

/* Kernel that computes factorial for each element of the input array */
__global__ void compute_factorials(const unsigned int *nums,
                                   unsigned long long *results,
                                   int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int val = nums[idx];
    unsigned long long res = factorial(val);

    results[idx] = res;

    /* Device printf for demonstration */
    printf("Thread %d: factorial(%u) = %llu\n", idx, val, res);
}

int main(void) {
    /* Example input */
    const int N = 3;
    unsigned int h_nums[N] = {5, 7, 9};
    unsigned long long h_results[N] = {0};

    /* Device pointers */
    unsigned int *d_nums = NULL;
    unsigned long long *d_results = NULL;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_nums, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, N * sizeof(unsigned long long)));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_nums, h_nums, N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    /* Set device stack size to allow recursion */
    size_t stackSize = 1024; /* bytes */
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    compute_factorials<<<blocks, threadsPerBlock>>>(d_nums, d_results, N);

    /* Synchronize and check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back to host */
    CUDA_CHECK(cudaMemcpy(h_results, d_results, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    /* Print results from host */
    printf("\nHost received results:\n");
    for (int i = 0; i < N; ++i) {
        printf("factorial(%u) = %llu\n", h_nums[i], h_results[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_nums));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}
