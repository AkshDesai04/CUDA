```cuda
/* 
Add a comment to your program explaining what 'concurrent kernel execution' means and how it can improve performance.

My thoughts:
- The aim of this program is to illustrate the concept of concurrent kernel execution in CUDA and explain how it can improve performance.
- Concurrent kernel execution refers to the ability of a GPU to run multiple kernels simultaneously (or overlap their execution) on the same device, as long as there are enough resources (SMs, registers, shared memory, etc.) to support them.
- When kernels are launched in different CUDA streams, the runtime schedules them in a way that maximizes hardware utilization. If one kernel is idle on some streaming multiprocessors (SMs) while another kernel is ready to run, the GPU can start the second kernel on those idle SMs. This overlap reduces idle time and can lead to higher throughput.
- In this example, we launch two simple kernels (`kernelA` and `kernelB`) in two separate streams. Each kernel writes to a distinct output array. Because the kernels use different output buffers, there are no conflicts, allowing the GPU to run them concurrently. The comment in the code explains the concept and its benefit.
- After launching the kernels, we synchronize the default stream to ensure both kernels have finished before we copy the results back to the host and print a confirmation.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel that writes the global thread index into an array
__global__ void kernelA(int *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = idx;            // write index as data
}

// Another kernel that writes a different value (e.g., squared index)
__global__ void kernelB(int *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = idx * idx;      // write squared index as data
}

// Utility macro for error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void)
{
    const int N = 1 << 20;          // number of elements
    const int bytes = N * sizeof(int);

    // Allocate host memory
    int *h_outA = (int *)malloc(bytes);
    int *h_outB = (int *)malloc(bytes);
    if (!h_outA || !h_outB) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_outA, *d_outB;
    CUDA_CHECK(cudaMalloc((void **)&d_outA, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_outB, bytes));

    // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Define block and grid sizes
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernelA in stream1
    kernelA<<<gridSize, blockSize, 0, stream1>>>(d_outA, N);

    // Launch kernelB in stream2
    kernelB<<<gridSize, blockSize, 0, stream2>>>(d_outB, N);

    // Optional: check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for all kernels to finish
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_outA, d_outA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outB, d_outB, bytes, cudaMemcpyDeviceToHost));

    // Simple validation (just check first element)
    if (h_outA[0] == 0 && h_outB[0] == 0)
        printf("Concurrent kernels executed successfully.\n");
    else
        printf("Data mismatch.\n");

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_outA));
    CUDA_CHECK(cudaFree(d_outB));
    free(h_outA);
    free(h_outB);

    return 0;
}
```