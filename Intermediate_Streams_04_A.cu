/*
Create two streams and one `cudaEvent_t`.

The program's aim is to demonstrate the creation of two CUDA streams and a single cudaEvent_t. 
The thought process was to build a minimal yet complete example that:
1. Creates the required CUDA resources (streams and event).
2. Launches a simple dummy kernel on the first stream.
3. Records an event on that stream.
4. Makes the second stream wait for the event.
5. Launches another kernel on the second stream.
6. Synchronizes both streams, copies data back to the host, prints a small sample, and cleans up all resources.
This will showcase the use of streams and events in CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 256) {
        data[idx] = idx;
    }
}

int main() {
    // Create event
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate device memory
    const int N = 256;
    int *devData;
    cudaMalloc(&devData, N * sizeof(int));

    // Launch dummy kernel on stream1
    dummyKernel<<<1, N, 0, stream1>>>(devData);

    // Record event on stream1 after the kernel
    cudaEventRecord(event, stream1);

    // Make stream2 wait for the event
    cudaStreamWaitEvent(stream2, event, 0);

    // Launch dummy kernel on stream2
    dummyKernel<<<1, N, 0, stream2>>>(devData);

    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Copy data back to host
    int host[N];
    cudaMemcpy(host, devData, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first five results
    printf("First 5 results: %d %d %d %d %d\n",
           host[0], host[1], host[2], host[3], host[4]);

    // Clean up
    cudaFree(devData);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}