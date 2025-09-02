/*
A `cudaDeviceSynchronize()` is implicitly happening after each operation, making it fully synchronous. Time the total execution of the loop.

Thinking: 
- The task requires writing a CUDA C program that demonstrates that cudaDeviceSynchronize() is implicitly invoked after each kernel launch or memory copy, making the sequence of operations fully synchronous. 
- We need to time the total execution time of a loop that repeatedly launches a kernel. 
- To measure the time we will use CUDA events (cudaEvent_t). We record a start event, then perform the loop with kernel launches. We record an end event after the loop, synchronize on the end event, and query elapsed time. 
- Because there is an implicit synchronization after each kernel launch, we do not need to add an explicit cudaDeviceSynchronize() after each launch. The timing measurement will capture the time of all kernel launches. 
- For completeness, we will allocate a small array, fill it on the host, copy it to the device, launch a simple kernel that performs an element-wise addition, and run this kernel many times in a loop. 
- After timing, we will copy the result back to the host, print the elapsed time, and clean up. 
- The code is written in CUDA C (.cu) and can be compiled with nvcc. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const int LOOP_COUNT = 1000;    // Number of kernel launches

    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Prepare CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Loop: launch kernel repeatedly
    for (int i = 0; i < LOOP_COUNT; ++i) {
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        // No explicit cudaDeviceSynchronize(); implicit sync after each launch
    }

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Total execution time for %d kernel launches: %.3f ms\n",
           LOOP_COUNT, milliseconds);

    // Copy result back to host (optional)
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result (optional)
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) {
            errors++;
            if (errors < 10) {
                printf("Error at index %d: %f\n", i, h_c[i]);
            }
        }
    }
    if (errors == 0) {
        printf("Result verification passed.\n");
    } else {
        printf("Result verification failed with %d errors.\n", errors);
    }

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
