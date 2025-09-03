```cpp
/* 
Explain the difference between `cudaDeviceReset()` and `cudaDeviceSynchronize()` in comments.

The aim of this program is to illustrate, through a small CUDA example, the distinct roles
of `cudaDeviceSynchronize()` and `cudaDeviceReset()` and to explain their differences
in a detailed comment block. I reasoned that a minimal kernel launch followed by
the two API calls, interleaved with informative print statements, will effectively
demonstrate the semantics:
- `cudaDeviceSynchronize()` is a blocking call that waits until the device has finished
  all preceding tasks, allowing the host to safely read device results or proceed
  with dependent work. It does *not* reset the device state.
- `cudaDeviceReset()` destroys the current CUDA context, freeing all resources,
  resetting the device to its initial state. It implicitly synchronizes (it blocks
  until all prior work is complete), but it also clears memory, kernels, and
  other context-specific state. After a reset, the device is ready for a new
  context, but any pending operations are cancelled.

By embedding explanatory comments directly in the code, we satisfy the requirement
to explain the difference in comments while providing a working CUDA program
that demonstrates the effect of each call.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple kernel that sets each element of an array to its index
__global__ void setToIndex(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

int main() {
    const int N = 256;
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int*)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return -1;
    }

    // Launch kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    setToIndex<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // -------------------------------------------------------------
    // At this point, the kernel launch is asynchronous. The host
    // continues executing without waiting for the device to finish.
    // -------------------------------------------------------------

    // Example 1: Using cudaDeviceSynchronize()
    // ----------------------------------------
    // `cudaDeviceSynchronize()` blocks the host until the device has
    // completed all preceding tasks. After this call, it is safe to
    // copy data back from device to host or perform any dependent
    // host operations. It does not reset the device context or
    // free resources.
    printf("Calling cudaDeviceSynchronize() to wait for kernel completion...\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }
    printf("cudaDeviceSynchronize() returned. Device tasks are complete.\n");

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }
    printf("First element after sync: %d\n", h_arr[0]); // Should be 0

    // -------------------------------------------------------------
    // Example 2: Using cudaDeviceReset()
    // ------------------------------------
    // `cudaDeviceReset()` destroys the current CUDA context. This
    // implicitly waits for all pending work to finish, but it also
    // clears all device memory, kernels, and other context state.
    // After a reset, the device is in its initial state and ready
    // for a new context. It is typically called at the end of a
    // program to clean up resources.
    printf("Calling cudaDeviceReset() to destroy the CUDA context...\n");
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        // Even on failure, proceed to free host memory
        free(h_arr);
        return -1;
    }
    printf("cudaDeviceReset() returned. Device context has been reset.\n");

    // After reset, the device memory pointer d_arr is no longer valid.
    // We cannot free it or use it. Attempting to do so would be undefined.
    // We must also reallocate device memory if needed.

    // Clean up host memory
    free(h_arr);

    printf("Program completed successfully.\n");
    return 0;
}
```