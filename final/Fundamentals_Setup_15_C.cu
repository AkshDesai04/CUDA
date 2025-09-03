```c
/*
Add comments explaining that `cudaDeviceReset()` destroys all allocations and contexts on the current device.

Thinking:
- The goal is to demonstrate that calling cudaDeviceReset() cleans up every allocation, stream, event,
  and context that were created for the current CUDA device.  
- To make this clear, we will:
  1. Allocate some device memory.
  2. Launch a trivial kernel that writes to that memory.
  3. Copy the result back to host and print it.
  4. Call cudaDeviceReset().
  5. Show that the device memory is no longer valid after the reset (by attempting to reallocate
     the same memory size, which will succeed because the previous allocation was freed).
- We include thorough comments explaining each step, especially the effect of cudaDeviceReset().
- Error checking is added for all CUDA API calls to illustrate that after a reset the device
  is returned to its initial state and all previous contexts are gone.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes a value to each element of an array
__global__ void initArray(int *d_arr, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = val;
    }
}

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main(void) {
    const int N = 256;
    int *d_array = NULL;
    int *h_array = (int*)malloc(N * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_array, N * sizeof(int)));

    // Launch kernel to initialize array on device
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocks, threadsPerBlock>>>(d_array, 42, N);
    CHECK_CUDA(cudaGetLastError());  // Check kernel launch
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 5 elements to verify
    printf("First 5 elements after kernel launch: ");
    for (int i = 0; i < 5; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free host memory
    free(h_array);

    // ==========================================================
    // Explanation of cudaDeviceReset()
    // ==========================================================
    // cudaDeviceReset() destroys all allocations and contexts on the current device.
    // This means that any memory previously allocated with cudaMalloc, any CUDA streams,
    // events, kernels, or other resources tied to the device context will be released.
    // After calling cudaDeviceReset(), the device is in the same state as if it had
    // just been powered on: no device memory is allocated, no context exists.
    // It is often used at the end of a CUDA application to clean up all resources
    // and ensure that subsequent CUDA API calls start from a clean slate.
    // ==========================================================

    // Reset device, destroying all allocations and contexts
    CHECK_CUDA(cudaDeviceReset());

    // After reset, the previous device memory pointer d_array is no longer valid.
    // Attempting to use it would result in an error. To illustrate that the reset
    // has freed the memory, we can allocate the same amount of memory again.

    int *d_newArray = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_newArray, N * sizeof(int)));  // This should succeed

    // Clean up the newly allocated memory
    CHECK_CUDA(cudaFree(d_newArray));

    printf("Device reset successfully; all previous allocations and contexts destroyed.\n");
    return EXIT_SUCCESS;
}
```