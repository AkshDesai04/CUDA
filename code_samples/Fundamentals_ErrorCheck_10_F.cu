```cpp
/*
Aim: Create two timers to measure two different parts of your program sequentially.

Thought Process:
1. I want to illustrate how to use CUDA events to time GPU operations in two distinct stages.
2. The program will perform two simple kernels on a large array:
   a. An element‑wise addition of two vectors (Stage 1).
   b. An element‑wise scaling of the result by 2 (Stage 2).
3. For each stage, I create a pair of CUDA events (start & stop) and record the elapsed time after each kernel launch.
4. I also keep the code minimal: allocate host and device memory, launch kernels, copy back results, and clean up.
5. Error checking is done via a macro to simplify code and avoid clutter.
6. After execution, the program prints the elapsed time (in milliseconds) for each stage.

The final code is a complete, compilable CUDA C program (.cu file) that demonstrates two sequential timers.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel 1: Element‑wise addition of two vectors
__global__ void addKernel(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

// Kernel 2: Element‑wise scaling of a vector by 2
__global__ void scaleKernel(float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] *= 2.0f;
}

int main(void) {
    const int N = 1 << 20;            // Size of vectors (about 1 million elements)
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // Create CUDA events for timing
    cudaEvent_t start1, stop1, start2, stop2;
    CHECK_CUDA(cudaEventCreate(&start1));
    CHECK_CUDA(cudaEventCreate(&stop1));
    CHECK_CUDA(cudaEventCreate(&start2));
    CHECK_CUDA(cudaEventCreate(&stop2));

    // Copy input vectors to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* ---------------------------------------------------------------
       Stage 1: Element‑wise addition
       --------------------------------------------------------------- */
    CHECK_CUDA(cudaEventRecord(start1, 0));          // Record start of Stage 1

    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());                 // Check for launch errors

    CHECK_CUDA(cudaEventRecord(stop1, 0));           // Record stop of Stage 1

    /* ---------------------------------------------------------------
       Stage 2: Element‑wise scaling
       --------------------------------------------------------------- */
    CHECK_CUDA(cudaEventRecord(start2, 0));          // Record start of Stage 2

    scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, N);
    CHECK_CUDA(cudaGetLastError());                 // Check for launch errors

    CHECK_CUDA(cudaEventRecord(stop2, 0));           // Record stop of Stage 2

    // Wait for all kernels to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Compute elapsed times
    float ms1 = 0, ms2 = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms1, start1, stop1));
    CHECK_CUDA(cudaEventElapsedTime(&ms2, start2, stop2));

    // Print timing results
    printf("Stage 1 (Addition) elapsed time: %f ms\n", ms1);
    printf("Stage 2 (Scaling)  elapsed time: %f ms\n", ms2);

    // Verify a few results
    bool correct = true;
    for (int i = 0; i < 5; ++i) {
        float expected = (h_a[i] + h_b[i]) * 2.0f;
        if (fabs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    if (correct)
        printf("Result verification: SUCCESS\n");
    else
        printf("Result verification: FAILURE\n");

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start1));
    CHECK_CUDA(cudaEventDestroy(stop1));
    CHECK_CUDA(cudaEventDestroy(start2));
    CHECK_CUDA(cudaEventDestroy(stop2));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```