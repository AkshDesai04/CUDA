/*
Create a dependency between two streams. Launch a kernel on stream1. Record an event on stream1 after the kernel. Then, make stream2 wait for that event using `cudaStreamWaitEvent` before launching its own kernel. This ensures the second kernel only starts after the first one is finished.

Thoughts and approach:
- The program demonstrates stream dependency via events.
- Two CUDA streams (stream1 and stream2) are created.
- Kernel1 writes values into a device array (devData1).
- After kernel1 finishes in stream1, an event (event1) is recorded.
- stream2 is instructed to wait on event1 before executing Kernel2 using cudaStreamWaitEvent.
- Kernel2 reads devData1 and writes processed results into another device array (devData2).
- After both kernels have executed, we copy the results back to the host and verify that the values in devData2 are as expected, confirming that Kernel2 indeed ran after Kernel1.
- Simple error-checking macro (CHECK) is used to validate CUDA API calls.
- The program prints the input and output arrays for visual confirmation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Simple kernel that writes i * 10 into devOut
__global__ void kernel1(int *devOut, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        devOut[idx] = idx * 10;
    }
}

// Kernel2 reads from devIn and writes devOut as devIn + 5
__global__ void kernel2(const int *devIn, int *devOut, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        devOut[idx] = devIn[idx] + 5;
    }
}

int main(void) {
    const int N = 16;
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int h_input[N], h_output[N];

    // Allocate device memory
    int *devData1, *devData2;
    CHECK(cudaMalloc((void **)&devData1, size));
    CHECK(cudaMalloc((void **)&devData2, size));

    // Create two streams
    cudaStream_t stream1, stream2;
    CHECK(cudaStreamCreate(&stream1));
    CHECK(cudaStreamCreate(&stream2));

    // Create event
    cudaEvent_t event1;
    CHECK(cudaEventCreate(&event1));

    // Launch kernel1 in stream1
    int threadsPerBlock = 4;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(devData1, N);
    CHECK(cudaGetLastError());

    // Record event after kernel1
    CHECK(cudaEventRecord(event1, stream1));

    // Make stream2 wait for event1
    CHECK(cudaStreamWaitEvent(stream2, event1, 0));

    // Launch kernel2 in stream2
    kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(devData1, devData2, N);
    CHECK(cudaGetLastError());

    // Synchronize streams to ensure all work is done
    CHECK(cudaStreamSynchronize(stream1));
    CHECK(cudaStreamSynchronize(stream2));

    // Copy result back to host
    CHECK(cudaMemcpy(h_input, devData1, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_output, devData2, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Kernel1 output (devData1):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    printf("Kernel2 output (devData2) (should be devData1 + 5):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Clean up
    CHECK(cudaEventDestroy(event1));
    CHECK(cudaStreamDestroy(stream1));
    CHECK(cudaStreamDestroy(stream2));
    CHECK(cudaFree(devData1));
    CHECK(cudaFree(devData2));

    return 0;
}
