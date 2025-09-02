```cuda
/*
Use `cudaEvent`s to time only the kernel execution time.

Thought Process:
1. I need a minimal yet complete CUDA program that demonstrates measuring kernel execution time.
2. The program will perform a simple vector addition (or any lightweight kernel) so that the
   focus remains on the timing mechanism rather than the computation itself.
3. I will use CUDA runtime API for memory allocation, kernel launch, and event handling.
4. Timing with `cudaEvent`:
   - Create two events: startEvent and stopEvent.
   - Record the start event just before launching the kernel.
   - Record the stop event immediately after the kernel launch.
   - Synchronize on the stop event to ensure the kernel has finished.
   - Compute elapsed time using `cudaEventElapsedTime`.
5. Since the requirement is to time *only* the kernel execution, I will not include
   host-side memory transfer times in the measurement.
6. The program will:
   - Allocate host memory for two input vectors and one output vector.
   - Allocate device memory.
   - Copy inputs to device.
   - Run the kernel with event timing.
   - Copy result back to host.
   - Verify correctness (optional).
   - Clean up resources.
7. I'll add error checking via a simple macro to keep the code concise.
8. Finally, the entire source code will be written in a single .cu file and the only
   output produced by this assistant will be that file content.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                            \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",  \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                        \
    }

// Simple element-wise vector addition kernel
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    // Kernel launch
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Optional verification
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            printf("Verification failed at index %d: %f != %f + %f\n",
                   i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }
    if (success) {
        printf("Verification PASSED.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```