/*
Can a stream wait on an event that has not yet been recorded? (Yes, it will wait until it is recorded and completed).

Thinking:
To demonstrate this behavior in CUDA, we set up two streams: streamA and streamB. 
streamA launches a simple kernel that performs an operation on an array and then records an event once that kernel finishes.
streamB, before the event has been recorded, is told to wait on that event. Because the event hasn't been recorded yet, streamB will block
until the event is recorded and the preceding work on streamA completes. After the event is recorded, streamB can continue with its own kernel.
This program illustrates the fact that a stream will indeed wait on an event that is recorded later, waiting for both the recording and the
completion of any prior operations associated with that event. The program uses cudaMemcpyAsync and asynchronous kernels to keep the
execution pipeline full and shows the correct synchronization.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel: add 1.0f to each element
__global__ void addOne(float *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] += 1.0f;
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    float *h_arr = (float*)malloc(size);
    for (int i = 0; i < N; ++i)
        h_arr[i] = 0.0f;

    float *d_arr = NULL;
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Create two streams
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // Create an event (not yet recorded)
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Launch kernel on streamA
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOne<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(d_arr, N);

    // Make streamB wait on event (event not yet recorded)
    cudaStreamWaitEvent(streamB, event, 0);

    // Record event on streamA after kernel completion
    cudaEventRecord(event, streamA);

    // After the event has been recorded and completed, streamB can continue.
    // Launch another kernel on streamB (same operation for demonstration)
    addOne<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(d_arr, N);

    // Wait for all work to complete
    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    // Copy result back
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Verify result: each element should be 2.0f
    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        if (h_arr[i] != 2.0f)
        {
            printf("Mismatch at index %d: %f\n", i, h_arr[i]);
            ok = false;
            break;
        }
    }

    if (ok)
        printf("Success: All elements are 2.0f.\n");
    else
        printf("Failure.\n");

    // Cleanup
    cudaEventDestroy(event);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
