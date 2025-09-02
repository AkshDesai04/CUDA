```cuda
/********************************************************************
 * Aim of the program:
 * Make sure you are using pinned host memory. The overlap will not
 * occur with pageable memory.
 *
 * Thinking:
 * 1. The objective is to demonstrate that using pinned (page-locked)
 *    host memory allows CUDA to perform asynchronous memory copies
 *    that can overlap with kernel execution. If we used pageable
 *    memory, cudaMemcpyAsync would fall back to a synchronous copy
 *    and no overlap would be observed.
 *
 * 2. We allocate a host array using cudaMallocHost, which guarantees
 *    the memory is page-locked. We also allocate device memory with
 *    cudaMalloc.
 *
 * 3. We create a single CUDA stream and two cudaEvent objects to
 *    time the operations. We then perform the following steps:
 *       a) Asynchronously copy data from host to device.
 *       b) Launch a simple kernel that increments each element.
 *       c) Asynchronously copy the result back to host.
 *
 *    Because the host memory is pinned, the copy operations can be
 *    overlapped with the kernel execution (subject to GPU capability
 *    and PCIe bandwidth). The events allow us to measure the total
 *    elapsed time and confirm that the asynchronous copy does not
 *    block the kernel.
 *
 * 4. We also include a sanity check that the result array on the host
 *    has indeed been incremented by one for each element.
 *
 * 5. Error checking macros are used throughout to keep the code
 *    clean and to ensure we catch any CUDA API errors immediately.
 *
 * 6. This program is self‑contained and can be compiled with nvcc:
 *       nvcc -arch=sm_60 -o pinned_overlap pinned_overlap.cu
 *
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that adds 1.0f to each element of an array
__global__ void add_one(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main(void) {
    const size_t N = 1 << 24;          // 16M elements
    const size_t bytes = N * sizeof(float);

    // Allocate pinned host memory
    float *h_data = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&h_data, bytes));

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create events to time the overlap
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Asynchronously copy data from host to device
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes,
                               cudaMemcpyHostToDevice, stream));

    // Launch kernel to add one to each element
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());

    // Asynchronously copy the result back to host
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, bytes,
                               cudaMemcpyDeviceToHost, stream));

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, stream));

    // Wait for all operations in the stream to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Calculate elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Elapsed time (ms): %f\n", ms);

    // Verify results
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabsf(h_data[i] - ((float)i + 1.0f)) > 1e-5f) {
            printf("Mismatch at index %zu: got %f, expected %f\n",
                   i, h_data[i], ((float)i + 1.0f));
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Result verification PASSED.\n");
    } else {
        printf("Result verification FAILED.\n");
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));

    return 0;
}
```