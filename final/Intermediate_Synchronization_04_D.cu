```cuda
/* 
Use other atomic functions like `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch`.

Thinking:
- The goal is to demonstrate the usage of several atomic operations provided by CUDA that are not the most common ones (`atomicAdd`, `atomicInc`). 
- We will create a small kernel that performs each of the requested atomic operations on device memory.
- `atomicSub` will decrement a shared counter.
- `atomicMin` and `atomicMax` will maintain the minimum and maximum values found in an input array.
- `atomicExch` will replace the value at each index with a constant (here 100) and can be used to retrieve the old value if desired.
- The host code will allocate and initialize the necessary arrays, launch the kernel, copy the results back, and print them to verify correctness.
- Simple error-checking macros will be used for clarity.
*/

#include <stdio.h>
#include <cuda.h>

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Kernel that demonstrates atomicSub, atomicMin, atomicMax, atomicExch
__global__ void atomicDemoKernel(int *counter,
                                 int *minVal,
                                 int *maxVal,
                                 int *exchange,
                                 const int *input,
                                 int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // atomicSub: decrement the shared counter by 1
    atomicSub(counter, 1);

    // atomicMin: update global minimum with input[idx]
    atomicMin(minVal, input[idx]);

    // atomicMax: update global maximum with input[idx]
    atomicMax(maxVal, input[idx]);

    // atomicExch: replace exchange[idx] with 100, returning old value
    int oldVal = atomicExch(&exchange[idx], 100);
    // (optional) we could store oldVal somewhere, but for this demo we ignore it
}

int main(void)
{
    const int N = 1024;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host memory
    int *h_input = (int *)malloc(N * sizeof(int));
    int *h_exchange = (int *)malloc(N * sizeof(int));
    int h_counter, h_minVal, h_maxVal;

    // Initialize input array with values 0..N-1
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
        h_exchange[i] = -1; // initial dummy value
    }

    // Initial counter value: N (since each thread will decrement by 1)
    h_counter = N;
    // Set minVal to a large number, maxVal to a small number
    h_minVal = INT_MAX;
    h_maxVal = INT_MIN;

    // Device memory
    int *d_input, *d_exchange, *d_counter, *d_minVal, *d_maxVal;
    CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_exchange, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_minVal, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_maxVal, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exchange, h_exchange, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_minVal, &h_minVal, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_maxVal, &h_maxVal, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    atomicDemoKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_counter,
                                                 d_minVal,
                                                 d_maxVal,
                                                 d_exchange,
                                                 d_input,
                                                 N);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_minVal, d_minVal, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxVal, d_maxVal, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_exchange, d_exchange, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Final counter (should be 0): %d\n", h_counter);
    printf("Minimum value in array: %d\n", h_minVal);
    printf("Maximum value in array: %d\n", h_maxVal);
    // Verify that all exchange values are now 100
    int allExchanged = 1;
    for (int i = 0; i < N; ++i) {
        if (h_exchange[i] != 100) {
            allExchanged = 0;
            break;
        }
    }
    printf("All exchange values replaced with 100: %s\n",
           allExchanged ? "Yes" : "No");

    // Clean up
    free(h_input);
    free(h_exchange);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_exchange));
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_minVal));
    CUDA_CHECK(cudaFree(d_maxVal));

    return 0;
}
```