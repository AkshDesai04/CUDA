```c
/*
Pass a pointer to a struct or class as the `user_data` to give the callback function context about the work that was completed.

The goal of this CUDA program is to demonstrate how to pass contextual data to a
stream callback via a pointer to a user-defined struct. In CUDA's runtime API, 
`cudaStreamAddCallback` allows a user to register a function that will be invoked
once all preceding operations on a stream have finished. The callback receives a
`void *userData` pointer which the programmer can set to anything, typically a
pointer to a struct that holds data the callback needs.

In this example we:

1. Define a simple struct `WorkContext` that contains some data about the
   work performed on the GPU.
2. Allocate a small piece of device memory and launch a trivial kernel that
   writes a value to that memory.
3. Register a callback on the stream, passing a pointer to a `WorkContext`
   instance as the user data.
4. In the callback we cast the `void *` back to `WorkContext *` and print the
   stored values, thereby demonstrating that the callback has context.

The code also includes basic error checking for CUDA API calls and uses
`cudaDeviceSynchronize` to ensure that all operations complete before the
program exits.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Struct that will hold context data for the callback
typedef struct {
    int   deviceResult;   // Value written by the kernel
    float timingSeconds;  // Example timing info
} WorkContext;

// Simple kernel that writes a value to device memory
__global__ void write_value_kernel(int *d_arr, int val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        d_arr[0] = val;
    }
}

// Callback function that receives the context pointer
void CUDART_CB stream_callback(cudaStream_t stream, cudaError_t status, void *userData) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Stream callback error: %s\n", cudaGetErrorString(status));
        return;
    }

    // Cast the userData back to our context struct
    WorkContext *ctx = (WorkContext *)userData;

    printf("[Callback] Device result = %d\n", ctx->deviceResult);
    printf("[Callback] Timing (s)   = %f\n", ctx->timingSeconds);
}

int main(void) {
    const int val_to_write = 42;
    const int array_size = 1;
    int *d_array = NULL;
    WorkContext ctx;
    ctx.deviceResult = 0;  // will be filled after kernel
    ctx.timingSeconds = 0.0f;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_array, array_size * sizeof(int)));

    // Create a stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Launch kernel on the stream
    write_value_kernel<<<1, 1, 0, stream>>>(d_array, val_to_write);

    // Capture start time
    clock_t start = clock();

    // Register callback, passing pointer to ctx as userData
    CUDA_CHECK(cudaStreamAddCallback(stream, stream_callback, (void *)&ctx, 0));

    // Wait for kernel to finish so we can copy result
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Capture end time
    clock_t end = clock();
    ctx.timingSeconds = ((float)(end - start)) / CLOCKS_PER_SEC;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&ctx.deviceResult, d_array, sizeof(int), cudaMemcpyDeviceToHost));

    // At this point the callback should have already executed and printed its output.
    // We can also print from main if desired
    printf("[Main] Device result after copy = %d\n", ctx.deviceResult);
    printf("[Main] Timing measured (s)    = %f\n", ctx.timingSeconds);

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
```