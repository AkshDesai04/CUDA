```cuda
/*
Aim: Ensure you call `cudaEventDestroy` to clean up the event handles.

Thinking:
- The task is to write a CUDA C program that demonstrates proper use of CUDA events
  for timing, and to ensure that the events are destroyed at the end of the program.
- The program will perform a simple vector addition on the GPU.
- Steps:
  1. Allocate host vectors and initialize them.
  2. Allocate device vectors.
  3. Copy host data to device.
  4. Create two CUDA events: start and stop.
  5. Record the start event.
  6. Launch the vector addition kernel.
  7. Record the stop event.
  8. Synchronize on the stop event to ensure completion.
  9. Calculate elapsed time between events.
 10. Copy result back to host and verify correctness.
 11. Destroy the events with `cudaEventDestroy`.
 12. Free device and host memory.
- I will include a simple error-checking macro to simplify CUDA API calls.
- The program will be self-contained, compile with `nvcc`, and will output the elapsed
  time and a basic validation message.
- The critical part requested by the prompt is the explicit call to `cudaEventDestroy`
  for both event handles before program termination.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Simple macro for error checking */
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

/* Kernel: vector addition */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Record start event */
    CHECK_CUDA(cudaEventRecord(start, 0));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    /* Record stop event */
    CHECK_CUDA(cudaEventRecord(stop, 0));

    /* Wait for the stop event to complete */
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Calculate elapsed time */
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Vector addition took %f ms.\n", milliseconds);

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int correct = 1;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Result verification: SUCCESS\n");
    } else {
        printf("Result verification: FAILURE\n");
    }

    /* Clean up: destroy events */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```