/*
Write a program that allocates memory on the GPU, then calls `cudaDeviceReset()`, and then attempts to use or free that memory. Observe the error that occurs.

The idea is to demonstrate that after a device reset, any previously allocated device memory becomes invalid. All handles to that memory are no longer valid, and any attempt to use or free them should produce a CUDA error. To show this, we will:

1. Allocate a small array on the GPU using cudaMalloc.
2. Launch a simple kernel that writes to this memory to confirm it works initially.
3. Call cudaDeviceReset() which destroys all allocations and resets the device state.
4. After the reset, we will:
   a. Attempt to launch the same kernel again on the same pointer. This should fail because the pointer is no longer valid.
   b. Attempt a cudaMemcpy from device to host using the stale pointer. This should also fail.
   c. Attempt to free the stale pointer with cudaFree. This should again produce an error.

For each CUDA API call, we will check the returned error code and print it, so we can see the exact error messages. The expected errors are usually `cudaErrorInvalidDevicePointer` or `cudaErrorInvalidValue`, depending on the operation.

The code below is a minimal, self‑contained CUDA C program that performs these steps and prints the results. Compile with `nvcc` and run on a machine with an active CUDA device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA errors */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
        } else {                                                            \
            printf("CUDA call %s succeeded.\n", #call);                     \
        }                                                                   \
    } while (0)

/* Simple kernel that writes thread indices to an array */
__global__ void write_kernel(int *d_arr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

int main() {
    const int N = 10;
    const size_t bytes = N * sizeof(int);
    int *d_ptr = NULL;

    /* Step 1: Allocate memory on the device */
    printf("Allocating %zu bytes on the device.\n", bytes);
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, bytes));

    /* Step 2: Launch kernel to initialize memory */
    printf("Launching kernel to initialize device memory.\n");
    int threadsPerBlock = 32;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    write_kernel<<<blocks, threadsPerBlock>>>(d_ptr, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Step 3: Reset the device */
    printf("Resetting the CUDA device.\n");
    CHECK_CUDA(cudaDeviceReset());

    /* At this point, d_ptr is no longer valid. Any further use should error. */

    /* Step 4a: Attempt to launch kernel again using stale pointer */
    printf("Attempting to launch kernel with stale pointer after reset.\n");
    write_kernel<<<blocks, threadsPerBlock>>>(d_ptr, N);
    CHECK_CUDA(cudaGetLastError());  // Expect error
    CHECK_CUDA(cudaDeviceSynchronize());  // Might also error

    /* Step 4b: Attempt cudaMemcpy from stale device pointer */
    printf("Attempting cudaMemcpy from stale device pointer after reset.\n");
    int host_arr[N];
    CHECK_CUDA(cudaMemcpy(host_arr, d_ptr, bytes, cudaMemcpyDeviceToHost));

    /* Step 4c: Attempt to free stale pointer */
    printf("Attempting cudaFree on stale pointer after reset.\n");
    CHECK_CUDA(cudaFree(d_ptr));

    /* Clean up: ensure device is reset before exiting */
    printf("Final device reset before exit.\n");
    CHECK_CUDA(cudaDeviceReset());

    printf("Program finished.\n");
    return 0;
}
