```cuda
/* 
Aim of the program: Can you reverse an array using only atomic swaps in global memory? Compare the performance. 

Thinking: 
1. We need a CUDA program that demonstrates array reversal using only atomic operations (atomicExch) on global memory, and compares its performance with a conventional (non-atomic) reverse. 
2. The atomic approach will involve one thread per swap: for each index i < N/2, a thread atomically swaps element i with element N-1-i. 
3. Since atomicExch is atomic per memory location, we can perform two atomicExch calls per swap: 
   - temp = atomicExch(&a[i], a[N-1-i]); 
   - atomicExch(&a[N-1-i], temp); 
   This sequence ensures that each variable is updated atomically, though the pair is not swapped in a single atomic transaction. 
4. The non-atomic approach uses simple register-based swaps (thread reads both values into registers, writes back). This is faster due to no atomic overhead. 
5. We'll allocate an array on host, fill it with sequential values, copy to device, run both kernels multiple times, measure execution time using cudaEvent, then verify correctness. 
6. We'll use block size 256, grid size determined by ceil(N/2 / 256). We'll run each kernel 10 times to average out jitter. 
7. Include proper error checking macros, free memory, and output times. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                      \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

__global__ void atomicSwapKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n / 2;
    if (idx < half) {
        int opposite = n - 1 - idx;
        // First atomic exchange: store old value of d_arr[idx] into temp, set d_arr[idx] to d_arr[opposite]
        int temp = atomicExch(&d_arr[idx], d_arr[opposite]);
        // Second atomic exchange: set d_arr[opposite] to old value of d_arr[idx]
        atomicExch(&d_arr[opposite], temp);
    }
}

__global__ void simpleSwapKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n / 2;
    if (idx < half) {
        int opposite = n - 1 - idx;
        // Simple register swap
        int temp = d_arr[idx];
        d_arr[idx] = d_arr[opposite];
        d_arr[opposite] = temp;
    }
}

void initializeArray(int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = i;
    }
}

int verifyReverse(const int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] != n - 1 - i) {
            printf("Mismatch at index %d: got %d, expected %d\n", i, arr[i], n - 1 - i);
            return 0;
        }
    }
    return 1;
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);
    int *h_arr = (int *)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    initializeArray(h_arr, N);

    int *d_arr_atomic, *d_arr_simple;
    CHECK(cudaMalloc((void **)&d_arr_atomic, size));
    CHECK(cudaMalloc((void **)&d_arr_simple, size));

    // Copy data to device
    CHECK(cudaMemcpy(d_arr_atomic, h_arr, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_arr_simple, h_arr, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    int blocks = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    // Timing variables
    cudaEvent_t start, stop;
    float time_atomic = 0.0f, time_simple = 0.0f;
    int trials = 10;

    // Prepare events
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up and timing for atomic kernel
    for (int t = 0; t < trials; ++t) {
        CHECK(cudaMemcpy(d_arr_atomic, h_arr, size, cudaMemcpyHostToDevice)); // reset
        CHECK(cudaEventRecord(start, 0));
        atomicSwapKernel<<<blocks, threadsPerBlock>>>(d_arr_atomic, N);
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        time_atomic += ms;
    }
    time_atomic /= trials;

    // Warm-up and timing for simple kernel
    for (int t = 0; t < trials; ++t) {
        CHECK(cudaMemcpy(d_arr_simple, h_arr, size, cudaMemcpyHostToDevice)); // reset
        CHECK(cudaEventRecord(start, 0));
        simpleSwapKernel<<<blocks, threadsPerBlock>>>(d_arr_simple, N);
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        time_simple += ms;
    }
    time_simple /= trials;

    // Copy results back to host for verification
    CHECK(cudaMemcpy(h_arr, d_arr_atomic, size, cudaMemcpyDeviceToHost));
    int ok_atomic = verifyReverse(h_arr, N);

    CHECK(cudaMemcpy(h_arr, d_arr_simple, size, cudaMemcpyDeviceToHost));
    int ok_simple = verifyReverse(h_arr, N);

    printf("Performance (average over %d trials):\n", trials);
    printf("  Atomic swap kernel:  %.3f ms\n", time_atomic);
    printf("  Simple swap kernel:  %.3f ms\n", time_simple);
    printf("Results verified: Atomic = %s, Simple = %s\n",
           ok_atomic ? "PASS" : "FAIL",
           ok_simple ? "PASS" : "FAIL");

    // Cleanup
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_arr_atomic));
    CHECK(cudaFree(d_arr_simple));
    free(h_arr);

    return 0;
}
```