/*
Aim: Create two streams and one `cudaEvent_t`.

Thinking:
The task is to write a minimal CUDA C program that demonstrates the creation of two CUDA streams and a single CUDA event. The goal is to show how to allocate the streams, create the event, use them for asynchronous kernel launches, record the event on one stream, and then synchronize or measure elapsed time. The program will:
1. Define a simple kernel that performs an element-wise addition.
2. Allocate memory on the device for input and output arrays.
3. Create two streams (`stream1` and `stream2`) using `cudaStreamCreate`.
4. Create one event (`event`) using `cudaEventCreate`.
5. Launch the kernel twice: once on each stream.
6. Record the event on the second stream after the second kernel launch.
7. Synchronize the event to ensure the second kernel has finished.
8. Measure the elapsed time between the start of the first kernel and the event completion.
9. Clean up by freeing device memory, destroying streams and the event.
10. Use a helper macro `checkCuda` to wrap CUDA API calls and report errors.

The program is self-contained and compiles as a `.cu` file. It demonstrates the basic usage of CUDA streams and events in an asynchronous execution context.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA API return status */
#define checkCuda(call)                                            \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

/* Simple element-wise addition kernel */
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const int N = 1 << 20;            // 1M elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc((void**)&d_a, size));
    checkCuda(cudaMalloc((void**)&d_b, size));
    checkCuda(cudaMalloc((void**)&d_c, size));

    /* Copy input data to device */
    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Create two streams */
    cudaStream_t stream1, stream2;
    checkCuda(cudaStreamCreate(&stream1));
    checkCuda(cudaStreamCreate(&stream2));

    /* Create an event */
    cudaEvent_t event;
    checkCuda(cudaEventCreate(&event));

    /* Launch kernel on stream1 */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, N);
    checkCuda(cudaGetLastError());

    /* Launch kernel on stream2 */
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_a, d_b, d_c, N);
    checkCuda(cudaGetLastError());

    /* Record the event after the second kernel launch */
    checkCuda(cudaEventRecord(event, stream2));

    /* Wait for event to complete (ensures stream2 has finished) */
    checkCuda(cudaEventSynchronize(event));

    /* Measure elapsed time from start to event completion */
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    /* Record start before launching kernels */
    checkCuda(cudaEventRecord(start, 0));
    /* For demonstration, we could launch more work here, but we keep it simple */

    /* Record stop after event has completed */
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Elapsed time: %f ms\n", milliseconds);

    /* Copy result back to host (though result is overwritten by second kernel) */
    checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify result (simple check) */
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Result %s\n", correct ? "correct" : "incorrect");

    /* Clean up */
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaEventDestroy(event));
    checkCuda(cudaStreamDestroy(stream1));
    checkCuda(cudaStreamDestroy(stream2));
    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
