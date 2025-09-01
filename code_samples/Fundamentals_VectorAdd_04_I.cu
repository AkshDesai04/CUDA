```cuda
/*
Time the subtraction kernel. Is it significantly different from addition?

The aim of this program is to measure and compare the execution time of a simple
vector addition kernel and a vector subtraction kernel on the GPU.  
To achieve this, the program:

1. Allocates input vectors A and B with a large number of elements (e.g., 1
   million floats).
2. Initializes these vectors with deterministic values so that the result
   can be verified.
3. Implements two device kernels:
   - `vecAdd` performs element‑wise addition: C[i] = A[i] + B[i].
   - `vecSub` performs element‑wise subtraction: C[i] = A[i] - B[i].
4. Uses CUDA events (`cudaEvent_t`) to time each kernel launch precisely:
   a. Record a start event.
   b. Launch the kernel.
   c. Record an end event.
   d. Synchronize and compute the elapsed time in milliseconds.
5. Copies the result back to host memory to ensure that the GPU actually
   executed the kernels (preventing any compiler optimizations that might
   skip kernel launches if the results were unused).
6. Prints the measured times for addition and subtraction, as well as the
   absolute and relative difference between them.
7. Cleans up all allocated memory and destroys CUDA events.

Because both kernels perform a single floating‑point operation per thread,
their computational patterns are identical. Therefore we expect the times
to be very close, with any difference attributable mainly to minor
differences in instruction scheduling or register usage. The program
demonstrates how to accurately time CUDA kernels using CUDA events
and provides a clear, reproducible comparison between two simple
arithmetic operations on the GPU.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 20)  // 1,048,576 elements

// Kernel for vector addition
__global__ void vecAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Kernel for vector subtraction
__global__ void vecSub(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] - B[idx];
}

// Helper function to check CUDA errors
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Allocate host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, N * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, N * sizeof(float)), "cudaMalloc d_C");

    // Copy data to device
    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy H2D B");

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t startAdd, stopAdd, startSub, stopSub;
    checkCuda(cudaEventCreate(&startAdd), "EventCreate startAdd");
    checkCuda(cudaEventCreate(&stopAdd), "EventCreate stopAdd");
    checkCuda(cudaEventCreate(&startSub), "EventCreate startSub");
    checkCuda(cudaEventCreate(&stopSub), "EventCreate stopSub");

    // Time vector addition
    checkCuda(cudaEventRecord(startAdd, 0), "EventRecord startAdd");
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaEventRecord(stopAdd, 0), "EventRecord stopAdd");
    checkCuda(cudaEventSynchronize(stopAdd), "EventSynchronize stopAdd");

    // Time vector subtraction
    checkCuda(cudaEventRecord(startSub, 0), "EventRecord startSub");
    vecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaEventRecord(stopSub, 0), "EventRecord stopSub");
    checkCuda(cudaEventSynchronize(stopSub), "EventSynchronize stopSub");

    // Compute elapsed times
    float msAdd = 0.0f, msSub = 0.0f;
    checkCuda(cudaEventElapsedTime(&msAdd, startAdd, stopAdd), "ElapsedTime Add");
    checkCuda(cudaEventElapsedTime(&msSub, startSub, stopSub), "ElapsedTime Sub");

    // Copy result back to host to ensure kernels executed
    checkCuda(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy D2H C");

    // Print timings
    printf("Vector Addition Time:   %6.3f ms\n", msAdd);
    printf("Vector Subtraction Time:%6.3f ms\n", msSub);
    printf("Absolute Difference:    %6.3f ms\n", fabs(msAdd - msSub));
    printf("Relative Difference:    %6.3f %%\n",
           100.0f * fabs(msAdd - msSub) / ((msAdd + msSub) / 2.0f));

    // Clean up
    checkCuda(cudaEventDestroy(startAdd), "EventDestroy startAdd");
    checkCuda(cudaEventDestroy(stopAdd), "EventDestroy stopAdd");
    checkCuda(cudaEventDestroy(startSub), "EventDestroy startSub");
    checkCuda(cudaEventDestroy(stopSub), "EventDestroy stopSub");
    checkCuda(cudaFree(d_A), "Free d_A");
    checkCuda(cudaFree(d_B), "Free d_B");
    checkCuda(cudaFree(d_C), "Free d_C");
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```