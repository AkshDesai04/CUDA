/*
The sequence is: kernelA<<<..., stream1>>>();, cudaEventRecord(event, stream1);, cudaStreamWaitEvent(stream2, event, 0);, kernelB<<<..., stream2>>>();

Thinking:  
The goal is to demonstrate a simple CUDA program that follows the exact sequence described in the prompt.  
1. Two CUDA streams (`stream1` and `stream2`) are created.  
2. `kernelA` is launched on `stream1` to perform some work (e.g., initialize an array).  
3. An event is recorded on `stream1` after `kernelA` completes.  
4. `stream2` is made to wait for that event before launching `kernelB`.  
5. `kernelB` then performs subsequent work that depends on the results of `kernelA`.  
6. The program synchronizes, copies data back to the host, and verifies the result.  

The code includes error checking, simple kernel implementations, and a minimal verification on the host. It should compile with `nvcc` and run on any CUDA-capable device.  
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

__global__ void kernelA(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = idx; // simple initialization
    }
}

__global__ void kernelB(const int *a, int *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] * 2; // simple processing based on kernelA's output
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const int bytes = N * sizeof(int);

    // Host allocation
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Device allocation
    int *d_a = nullptr;
    int *d_b = nullptr;
    CHECK(cudaMalloc((void**)&d_a, bytes));
    CHECK(cudaMalloc((void**)&d_b, bytes));

    // Create streams and event
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    CHECK(cudaStreamCreate(&stream1));
    CHECK(cudaStreamCreate(&stream2));
    CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    // Launch kernelA on stream1
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernelA<<<blocks, threadsPerBlock, 0, stream1>>>(d_a, N);
    CHECK(cudaGetLastError());

    // Record event on stream1 after kernelA
    CHECK(cudaEventRecord(event, stream1));

    // Make stream2 wait for the event before launching kernelB
    CHECK(cudaStreamWaitEvent(stream2, event, 0));

    // Launch kernelB on stream2
    kernelB<<<blocks, threadsPerBlock, 0, stream2>>>(d_a, d_b, N);
    CHECK(cudaGetLastError());

    // Synchronize both streams
    CHECK(cudaStreamSynchronize(stream1));
    CHECK(cudaStreamSynchronize(stream2));

    // Copy results back to host
    CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_a[i] != i || h_b[i] != i * 2) {
            ok = false;
            if (i < 10) { // print a few failures
                printf("Mismatch at %d: a=%d, b=%d\n", i, h_a[i], h_b[i]);
            }
        }
    }
    if (ok) {
        printf("Success: All values verified.\n");
    } else {
        printf("Failure: Verification failed.\n");
    }

    // Cleanup
    free(h_a);
    free(h_b);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaEventDestroy(event));
    CHECK(cudaStreamDestroy(stream1));
    CHECK(cudaStreamDestroy(stream2));

    return EXIT_SUCCESS;
}
