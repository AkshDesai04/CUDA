```cpp
/*
Aim of the program: Wrap all cudaFree calls.

Thought Process:
- The goal is to encapsulate all calls to cudaFree within a wrapper function that
  checks the return status and reports errors.  This approach ensures that
  freeing memory is safe, that any CUDA errors are caught immediately, and
  that debugging information is available.
- We define a helper function `safeCudaFree(void* ptr)` that performs the
  cudaFree operation and then checks the returned cudaError_t.  If an error
  occurs, it prints an informative message and calls `cudaDeviceReset()` to
  cleanly shutdown the device.
- To make the wrapper convenient, a macro `CUDA_FREE(ptr)` is defined that
  simply calls `safeCudaFree(ptr)`.  This allows the user to replace every
  `cudaFree` call with `CUDA_FREE` in their code, ensuring consistency.
- A simple demo is provided: we allocate memory on the device, launch a tiny
  kernel that writes values into that memory, copy the results back, print
  them, and then free the device memory using the wrapper.  We free the memory
  twice intentionally to demonstrate that the wrapper reports an error on the
  second free attempt.
- Error handling in CUDA often returns an error after the operation.  By
  checking immediately after the call, we prevent cascading errors and
  provide clear diagnostics.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Wrapper function for cudaFree
void safeCudaFree(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing memory at %p: %s\n", ptr, cudaGetErrorString(err));
        // Reset device to clean up state
        cudaDeviceReset();
        // Exit program to avoid undefined behavior
        exit(EXIT_FAILURE);
    } else {
        printf("Successfully freed memory at %p\n", ptr);
    }
}

// Macro to simplify usage
#define CUDA_FREE(ptr) safeCudaFree(ptr)

// A simple kernel that writes its index into the array
__global__ void writeIndices(int* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx;
    }
}

int main(void) {
    const int N = 10;
    int *d_arr = NULL;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating memory: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch kernel
    int threadsPerBlock = 4;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    writeIndices<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        CUDA_FREE(d_arr);
        return EXIT_FAILURE;
    }

    // Copy results back to host
    int h_arr[N];
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy error: %s\n", cudaGetErrorString(err));
        CUDA_FREE(d_arr);
        return EXIT_FAILURE;
    }

    // Print results
    printf("Device array contents:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory using wrapper
    CUDA_FREE(d_arr);

    // Attempt to free again to demonstrate error handling
    printf("Attempting to free memory again (should trigger error)...\n");
    CUDA_FREE(d_arr); // This should print an error and exit

    // Reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
```