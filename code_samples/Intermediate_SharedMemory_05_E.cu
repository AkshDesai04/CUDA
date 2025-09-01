```cuda
/* 
Aim of the program: 
Compare the performance of the boundary-checked kernel on a perfect-multiple size (e.g. 512x512) vs. the original kernel without checks. Is there overhead?

Thinking:
- We need a simple kernel that can run on a grid where the total number of threads equals the number of elements (no threads are idle). 
- The operation will be trivial (increment each element) so the kernel runtime will be dominated by memory and launch overhead, not compute.
- Two kernels: 
    1. kernelAddChecked - each thread computes a linear index and checks if it is within bounds before writing. 
    2. kernelAddUnchecked - the same but without the bound check.
- Use a perfect multiple of the block size (512x512 elements, block 16x16). This means every thread has a valid element; the branch taken in the checked kernel should always succeed. 
- Measure execution time using cudaEvent timing for a single launch of each kernel. 
- Run both kernels multiple times to reduce noise and average the times. 
- Verify correctness by copying back the data and ensuring all elements have been incremented. 
- Print the average times and compute the overhead percentage. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(err)                                               \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);         \
        exit(EXIT_FAILURE);                                           \
    }

// Kernel with boundary check
__global__ void kernelAddChecked(float *data, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * N + x;
    if (idx < N * N) {
        data[idx] += 1.0f;
    }
}

// Kernel without boundary check
__global__ void kernelAddUnchecked(float *data, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * N + x;
    data[idx] += 1.0f; // No check
}

int main()
{
    const int N = 512;                     // Matrix dimension (perfect multiple)
    const int numElements = N * N;
    const size_t size = numElements * sizeof(float);

    // Allocate host memory
    float *h_data = (float*)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    // Initialize host data to zeros
    for (int i = 0; i < numElements; ++i) h_data[i] = 0.0f;

    // Allocate device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16);                 // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    const int numRuns = 100; // Number of runs for averaging
    float elapsedChecked = 0.0f;
    float elapsedUnchecked = 0.0f;

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run checked kernel multiple times
    for (int i = 0; i < numRuns; ++i) {
        // Reset device data to zeros
        CUDA_CHECK(cudaMemset(d_data, 0, size));

        CUDA_CHECK(cudaEventRecord(start));
        kernelAddChecked<<<gridDim, blockDim>>>(d_data, N);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        elapsedChecked += ms;
    }
    float avgChecked = elapsedChecked / numRuns;

    // Run unchecked kernel multiple times
    for (int i = 0; i < numRuns; ++i) {
        // Reset device data to zeros
        CUDA_CHECK(cudaMemset(d_data, 0, size));

        CUDA_CHECK(cudaEventRecord(start));
        kernelAddUnchecked<<<gridDim, blockDim>>>(d_data, N);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        elapsedUnchecked += ms;
    }
    float avgUnchecked = elapsedUnchecked / numRuns;

    // Copy result back to host to verify correctness
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify all elements are 1.0f
    bool correct = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_data[i] != 1.0f) {
            correct = false;
            printf("Mismatch at index %d: %f\n", i, h_data[i]);
            break;
        }
    }

    if (correct) {
        printf("Result verification: PASSED\n");
    } else {
        printf("Result verification: FAILED\n");
    }

    // Print timing results
    printf("\nAverage execution time over %d runs:\n", numRuns);
    printf("Boundary-checked kernel : %f ms\n", avgChecked);
    printf("Unchecked kernel         : %f ms\n", avgUnchecked);
    float overhead = ((avgChecked - avgUnchecked) / avgUnchecked) * 100.0f;
    printf("Overhead due to boundary check: %.2f%%\n", overhead);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
```