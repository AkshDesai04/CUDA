```cpp
/*
Aim of the program:
If you are using CUDA events, wrap `cudaEventCreate`, `cudaEventRecord`, and `cudaEventDestroy`.

Thinking:
The goal is to demonstrate a clean, reusable way to manage CUDA events by providing wrapper
functions around the three core CUDA event API calls. These wrappers will:
1. Call the underlying CUDA API.
2. Check for errors immediately and report them with meaningful context.
3. Optionally print diagnostic messages indicating the event name and action taken.

The program will:
- Allocate a simple vector addition problem on the host and device.
- Create two CUDA events (start and stop) using the wrapper `createEvent`.
- Record the start event before launching the kernel and the stop event afterward
  using the wrapper `recordEvent`.
- Synchronize on the stop event, measure elapsed time, and print it.
- Destroy the events with the wrapper `destroyEvent`.
- Clean up all allocated memory.

The wrapper functions provide a clear, consistent error handling strategy
and make the code easier to read and maintain. They also accept an optional
`name` parameter to label events in debug output.

The program is fully self‑contained, written in CUDA C, and can be compiled
with `nvcc`:
    nvcc -o cuda_events_example cuda_events_example.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Utility macro for CUDA error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s (%s)\n",      \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Wrapper for cudaEventCreate with optional name for diagnostics */
cudaEvent_t createEvent(const char* name)
{
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error creating event '%s': %s\n", name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Created event '%s'\n", name);
    return event;
}

/* Wrapper for cudaEventRecord with optional name and stream */
void recordEvent(cudaEvent_t event, cudaStream_t stream = 0, const char* name = "")
{
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error recording event '%s' on stream %d: %s\n",
                name, (int)stream, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Recorded event '%s' on stream %d\n", name, (int)stream);
}

/* Wrapper for cudaEventDestroy with optional name */
void destroyEvent(cudaEvent_t event, const char* name = "")
{
    cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error destroying event '%s': %s\n", name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Destroyed event '%s'\n", name);
}

/* Simple vector addition kernel */
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1 million elements
    const size_t size = N * sizeof(float);

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
        h_B[i] = (float)(2 * i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Create CUDA events */
    cudaEvent_t startEvent = createEvent("start");
    cudaEvent_t stopEvent  = createEvent("stop");

    /* Record the start event */
    recordEvent(startEvent, 0, "start");

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Record the stop event */
    recordEvent(stopEvent, 0, "stop");

    /* Synchronize to make sure kernel has finished */
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    /* Compute elapsed time */
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    printf("Kernel execution time: %f ms\n", milliseconds);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Destroy events */
    destroyEvent(startEvent, "start");
    destroyEvent(stopEvent,  "stop");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```