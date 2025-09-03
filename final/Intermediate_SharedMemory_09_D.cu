/*
Aim of the program (word to word):
Write a kernel where `output[i] = shared_mem[threadIdx.x * 2]`. Profile this. On modern GPUs, this may be handled by the cache, but on older ones it would cause 2-way conflicts.

Thinking:
- The kernel needs to use shared memory and perform a potentially conflicting read pattern.
- Each thread will write a known pattern to the shared memory so we can verify the output.
- The read pattern is `shared_mem[threadIdx.x * 2]`; for `blockDim.x` threads this requires at least `blockDim.x * 2` shared memory slots.
- To avoid out‑of‑bounds access, we allocate shared memory of size `blockDim.x * 2 * sizeof(float)`.
- We fill the shared memory using a simple loop that each thread participates in, ensuring all slots are written.
- After synchronizing, each thread reads its assigned element and writes it to global memory.
- In the host code, we allocate device output, launch the kernel, time it with CUDA events, copy back and print a few results.
- Error checking macros help catch launch or memory errors.
- The kernel will be profiled to illustrate the effect of this memory pattern on different GPUs.
*/

#include <stdio.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                              \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

__global__ void sharedMemKernel(float *output)
{
    // Allocate shared memory dynamically; size specified at kernel launch
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Fill shared memory with a known pattern
    // Each thread writes to multiple slots to ensure all slots are initialized
    for (int i = tid; i < blockSize * 2; i += blockSize)
    {
        shared_mem[i] = (float)i; // simple deterministic value
    }

    __syncthreads(); // Ensure all writes complete before reading

    // Read from the potentially conflicting address and write to global memory
    output[blockIdx.x * blockSize + tid] = shared_mem[tid * 2];
}

int main(void)
{
    const int threadsPerBlock = 256; // typical warp size multiple
    const int numBlocks = 32;        // enough work to measure
    const int totalThreads = threadsPerBlock * numBlocks;

    size_t outputSize = totalThreads * sizeof(float);

    // Allocate host memory
    float *h_output = (float *)malloc(outputSize);
    if (!h_output)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_output, outputSize));

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel with dynamic shared memory size
    size_t sharedMemBytes = threadsPerBlock * 2 * sizeof(float);
    sharedMemKernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(d_output);

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));

    // Print some results
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    // Clean up
    free(h_output);
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
