/*
Change the launch configuration to 8 blocks of 32 threads and verify the global IDs are still unique and correct.

Thinking:
- The goal is to demonstrate that with a launch configuration of 8 blocks each having 32 threads,
  the computed global thread IDs are unique and correctly span the range [0, 256).
- We'll allocate a device array of size 256 and have each thread write its computed global ID
  to the array element at the same index (i.e., arr[global_id] = global_id).
- After the kernel finishes, the host will copy the array back and verify that each element
  equals its index. This ensures uniqueness and correctness.
- We also provide a simple error-checking macro for CUDA calls and print a success or failure
  message based on the verification.
- No external dependencies are required; the program is self‑contained and can be compiled
  with `nvcc program.cu -o program`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that writes each thread's global ID to a device array
__global__ void write_global_ids(int *d_arr)
{
    // Compute global thread ID
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't write out of bounds (should not happen with correct launch config)
    if (global_id < blockDim.x * gridDim.x) {
        d_arr[global_id] = global_id;
    }
}

int main(void)
{
    const int blocks = 8;
    const int threads_per_block = 32;
    const int total_threads = blocks * threads_per_block;

    // Allocate device memory
    int *d_arr;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, total_threads * sizeof(int)));

    // Launch kernel with 8 blocks of 32 threads
    write_global_ids<<<blocks, threads_per_block>>>(d_arr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate host memory and copy back results
    int *h_arr = (int *)malloc(total_threads * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, total_threads * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify uniqueness and correctness
    bool success = true;
    for (int i = 0; i < total_threads; ++i) {
        if (h_arr[i] != i) {
            fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n", i, i, h_arr[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("All global IDs are unique and correct. Test passed.\n");
    } else {
        printf("Test failed. See error messages above.\n");
    }

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
