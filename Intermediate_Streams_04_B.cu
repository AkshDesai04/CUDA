```cuda
/*
Aim: The sequence is: kernelA<<<..., stream1>>>();, cudaEventRecord(event, stream1);, cudaStreamWaitEvent(stream2, event, 0);, kernelB<<<..., stream2>>>();

Thought Process:
1. The purpose of this example is to demonstrate how to enforce a dependency between two CUDA kernels that are launched in different streams using an event.
2. We create two CUDA streams: stream1 and stream2.
3. kernelA writes into a device array. After launching kernelA on stream1 we record an event in that stream.
4. We then make stream2 wait for that event, ensuring that kernelB will not start executing until kernelA has finished.
5. kernelB reads the output of kernelA, processes it, and writes the result to another device array.
6. Finally we copy the results back to host memory and verify correctness.
7. All CUDA API calls are checked for errors.
8. The code uses simple kernels (add and multiply) for clarity, but the pattern applies to any user kernels.
9. This example can be compiled with nvcc, e.g. `nvcc -arch=sm_70 dependency_example.cu -o dependency_example`.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    __func__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Simple kernel that adds a constant value to each element
__global__ void kernelA(float *d_out, float value, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = value; // For demonstration, just write the value
    }
}

// Simple kernel that multiplies each element by a constant
__global__ void kernelB(const float *d_in, float *d_out, float mul, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = d_in[idx] * mul;
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_result = (float*)malloc(bytes);
    if (!h_result) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    float *d_a, *d_b;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));

    // Create two streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Create an event
    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));

    // Launch kernelA on stream1
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernelA<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, 3.14f, N);
    CHECK_CUDA(cudaGetLastError());

    // Record event after kernelA completes in stream1
    CHECK_CUDA(cudaEventRecord(event, stream1));

    // Make stream2 wait for the event
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event, 0));

    // Launch kernelB on stream2; it depends on kernelA's output
    kernelB<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_a, d_b, 2.0f, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host; use stream2 so copy waits for kernelB
    CHECK_CUDA(cudaMemcpyAsync(h_result, d_b, bytes, cudaMemcpyDeviceToHost, stream2));

    // Synchronize to make sure all operations are finished
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    // Verify result (should be 3.14 * 2.0 = 6.28)
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (abs(h_result[i] - 6.28f) > 1e-4f) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at index %d: %f\n", i, h_result[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("All results correct.\n");
    } else {
        printf("Found %d mismatches.\n", errors);
    }

    // Clean up
    free(h_result);
    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return EXIT_SUCCESS;
}
```